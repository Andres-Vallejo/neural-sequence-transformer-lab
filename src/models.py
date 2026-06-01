import torch
from torch import nn


class FastTextLikeClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 96, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.2), nn.Linear(embed_dim, num_classes))

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        mask = (input_ids != 0).unsqueeze(-1).float()
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 96, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        _, (h, _) = self.encoder(emb)
        pooled = torch.cat([h[-2], h[-1]], dim=1)
        return self.classifier(pooled)


class SequenceAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 96, hidden_dim: int = 128, latent_dim: int = 96, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        _, h = self.encoder(emb)
        z = torch.tanh(self.to_latent(h[-1]))
        h0 = self.from_latent(z).unsqueeze(0)
        decoded, _ = self.decoder(emb, h0)
        return self.output(decoded)


class TransformerTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, max_length: int = 48, embed_dim: int = 128, depth: int = 3, num_heads: int = 4, mlp_dim: int = 256, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids) + self.position_embedding[:, :input_ids.size(1)]
        key_padding_mask = input_ids.eq(0)
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        mask = (~key_padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


def build_model(name: str, vocab_size: int, num_classes: int, max_length: int, cfg: dict):
    name = name.lower()
    if name == "embedding":
        return FastTextLikeClassifier(vocab_size, num_classes, **cfg)
    if name == "bilstm":
        return BiLSTMClassifier(vocab_size, num_classes, **cfg)
    if name == "autoencoder":
        return SequenceAutoencoder(vocab_size, **cfg)
    if name == "transformer":
        return TransformerTextClassifier(vocab_size, num_classes, max_length=max_length, **cfg)
    raise ValueError(f"Unknown model: {name}")
