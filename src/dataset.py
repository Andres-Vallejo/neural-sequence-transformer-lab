import re
from collections import Counter
from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"


def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


@dataclass
class Vocabulary:
    stoi: dict
    itos: list

    @classmethod
    def build(cls, texts, min_freq: int = 1):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        itos = [PAD, UNK, BOS, EOS] + sorted([tok for tok, freq in counter.items() if freq >= min_freq])
        stoi = {tok: idx for idx, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, max_length: int):
        ids = [self.stoi[BOS]] + [self.stoi.get(tok, self.stoi[UNK]) for tok in tokenize(text)] + [self.stoi[EOS]]
        ids = ids[:max_length]
        ids += [self.stoi[PAD]] * (max_length - len(ids))
        return ids


class TicketDataset(Dataset):
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


def get_dataloaders(path: str, text_column: str, label_column: str, max_length: int, batch_size: int):
    df = pd.read_csv(path)
    vocab = Vocabulary.build(df[text_column])
    labels = sorted(df[label_column].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}
    dataset = TicketDataset(df, vocab, label_map, text_column, label_column, max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        vocab,
        label_map,
    )
