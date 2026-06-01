import argparse
from pathlib import Path
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import yaml
from dataset import get_dataloaders
from models import build_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["embedding", "bilstm", "transformer"], default="transformer")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    _, _, test_loader, vocab, label_map = get_dataloaders(**cfg["data"])
    inv_labels = {idx: label for label, idx in label_map.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, len(vocab.itos), len(label_map), cfg["data"]["max_length"], cfg["models"][args.model]).to(device)
    checkpoint = torch.load(Path(cfg["outputs"]["checkpoint_dir"]) / f"{args.model}_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    y_true, y_pred = [], []
    for batch in test_loader:
        logits = model(batch["input_ids"].to(device))
        y_true.extend(batch["labels"].tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    labels = [inv_labels[i] for i in range(len(inv_labels))]
    metrics_dir = Path(cfg["outputs"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(labels)))
    report = classification_report(y_true, y_pred, labels=label_ids, target_names=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(metrics_dir / f"{args.model}_classification_report.csv")
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=label_ids), index=labels, columns=labels).to_csv(metrics_dir / f"{args.model}_confusion_matrix.csv")
    print(pd.DataFrame(report).T)


if __name__ == "__main__":
    main()
