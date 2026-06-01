import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

AG_NEWS_LABELS = {
    1: "world",
    2: "sports",
    3: "business",
    4: "science_technology",
}


def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9']+", str(text).lower())


def download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)


def load_ag_news(data_dir: str, train_url: str, test_url: str):
    root = Path(data_dir)
    train_path = root / "ag_news_train.csv"
    test_path = root / "ag_news_test.csv"
    download_if_missing(train_url, train_path)
    download_if_missing(test_url, test_path)
    columns = ["label_id", "title", "description"]
    train = pd.read_csv(train_path, names=columns)
    test = pd.read_csv(test_path, names=columns)
    for frame in (train, test):
        frame["label"] = frame["label_id"].map(AG_NEWS_LABELS)
        frame["text"] = frame["title"].fillna("") + " " + frame["description"].fillna("")
    return train[["text", "label"]], test[["text", "label"]]


@dataclass
class Vocabulary:
    stoi: dict
    itos: list

    @classmethod
    def build(cls, texts, min_freq: int = 2, max_tokens: int = 30000):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        tokens = [tok for tok, freq in counter.most_common(max_tokens) if freq >= min_freq]
        itos = [PAD, UNK, BOS, EOS] + tokens
        stoi = {tok: idx for idx, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, max_length: int):
        ids = [self.stoi[BOS]]
        ids.extend(self.stoi.get(tok, self.stoi[UNK]) for tok in tokenize(text))
        ids.append(self.stoi[EOS])
        ids = ids[:max_length]
        ids += [self.stoi[PAD]] * (max_length - len(ids))
        return ids


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: Vocabulary, label_map: dict, text_column: str, label_column: str, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.vocab = vocab
        self.label_map = label_map
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        input_ids = torch.tensor(self.vocab.encode(row[self.text_column], self.max_length), dtype=torch.long)
        label = torch.tensor(self.label_map[row[self.label_column]], dtype=torch.long)
        attention_mask = (input_ids != self.vocab.stoi[PAD]).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}


def sample_frame(frame: pd.DataFrame, max_samples: int | None, seed: int = 42):
    if max_samples and len(frame) > max_samples:
        return frame.sample(max_samples, random_state=seed).reset_index(drop=True)
    return frame.reset_index(drop=True)


def get_dataloaders(
    source: str,
    data_dir: str,
    train_url: str,
    test_url: str,
    text_column: str,
    label_column: str,
    max_length: int,
    batch_size: int,
    val_split: float = 0.1,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
):
    if source != "ag_news":
        raise ValueError(f"Unsupported source: {source}")

    train_frame, test_frame = load_ag_news(data_dir, train_url, test_url)
    train_frame = train_frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_size = int(len(train_frame) * val_split)
    val_frame = train_frame.iloc[:val_size]
    train_frame = train_frame.iloc[val_size:]
    train_frame = sample_frame(train_frame, max_train_samples)
    val_frame = sample_frame(val_frame, max_val_samples)
    test_frame = sample_frame(test_frame, max_test_samples)

    vocab = Vocabulary.build(train_frame[text_column])
    labels = sorted(train_frame[label_column].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}

    train_ds = TextDataset(train_frame, vocab, label_map, text_column, label_column, max_length)
    val_ds = TextDataset(val_frame, vocab, label_map, text_column, label_column, max_length)
    test_ds = TextDataset(test_frame, vocab, label_map, text_column, label_column, max_length)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        vocab,
        label_map,
    )
