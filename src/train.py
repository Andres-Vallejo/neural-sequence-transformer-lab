import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
import yaml
from dataset import get_dataloaders
from models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_config(name: str):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def run_epoch(model, loader, criterion, optimizer, device, autoencoder=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        if autoencoder:
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.reshape(-1))
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
        if not autoencoder:
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader.dataset), correct / total if total else None


@torch.no_grad()
def validate(model, loader, criterion, device, autoencoder=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids)
        if autoencoder:
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.reshape(-1))
        else:
            loss = criterion(outputs, labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset), correct / total if total else None


def improved(metric: float, best_metric: float, min_delta: float, mode: str) -> bool:
    if mode == "max":
        return metric > best_metric + min_delta
    return metric < best_metric - min_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["embedding", "bilstm", "autoencoder", "transformer"], default="transformer")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    train_loader, val_loader, _, vocab, label_map = get_dataloaders(**cfg["data"])
    device = device_from_config(cfg["training"].get("device", "auto"))
    model = build_model(args.model, len(vocab.itos), len(label_map), cfg["data"]["max_length"], cfg["models"][args.model]).to(device)
    autoencoder = args.model == "autoencoder"
    criterion = nn.CrossEntropyLoss(ignore_index=0) if autoencoder else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    monitor = cfg["training"]["early_stopping"]["monitor"]
    monitor_mode = "min" if autoencoder or monitor.endswith("loss") else "max"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=monitor_mode,
        factor=cfg["training"]["scheduler"]["factor"],
        patience=cfg["training"]["scheduler"]["patience"],
    )
    epochs = args.epochs or cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping"]["patience"]
    min_delta = cfg["training"]["early_stopping"]["min_delta"]
    ckpt_dir = Path(cfg["outputs"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -float("inf") if monitor_mode == "max" else float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, autoencoder)
        val_loss, val_acc = validate(model, val_loader, criterion, device, autoencoder)
        metric = val_loss if autoencoder or monitor == "val_loss" else val_acc
        scheduler.step(metric)
        current_lr = optimizer.param_groups[0]["lr"]
        print({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4), "train_acc": train_acc, "val_acc": val_acc, "lr": current_lr})
        if improved(metric, best_metric, min_delta, monitor_mode):
            best_metric = metric
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "vocab": vocab, "label_map": label_map, "model": args.model, "config": cfg, "best_metric": best_metric, "best_epoch": epoch}, ckpt_dir / f"{args.model}_best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print({"early_stopping": True, "epoch": epoch, "best_metric": best_metric, "monitor": monitor})
                break


if __name__ == "__main__":
    main()
