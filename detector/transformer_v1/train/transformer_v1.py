# =========================================
# 双 Transformer 提交缺陷分类器
# =========================================

import math
import os
import json
import warnings
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings(
    "ignore",
    message=(
        "The PyTorch API of nested tensors is in prototype stage and will change "
        "in the near future\\."
    ),
    category=UserWarning
)
print("Using device:", device)

DATA_PATH = "final_diff_dataset.csv"


df = pd.read_csv(DATA_PATH)
df = df[["PR_TITLE", "DIFF_CODE", "IS_REVERSED"]].dropna()

tokenizer = get_tokenizer("basic_english")

counter = Counter()
for text in df["PR_TITLE"]:
    counter.update(tokenizer(str(text)))

for text in df["DIFF_CODE"]:
    counter.update(tokenizer(str(text)))

vocab = Vocab(counter, specials=["<pad>", "<unk>"])
vocab.unk_index = vocab.stoi["<unk>"]
PAD_IDX = vocab.stoi["<pad>"]
print("Vocab size:", len(vocab))

class DiffDataset(Dataset):
    def __init__(self, df, vocab, max_title_len, max_diff_len):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_title_len = max_title_len
        self.max_diff_len = max_diff_len

    def encode(self, text, max_len):
        tokens = tokenizer(str(text))
        ids = [
            self.vocab.stoi.get(t, self.vocab.unk_index)
            for t in tokens
        ][:max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            self.encode(row["PR_TITLE"], self.max_title_len),
            self.encode(row["DIFF_CODE"], self.max_diff_len),
            torch.tensor(row["IS_REVERSED"], dtype=torch.float)
        )

    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    titles, diffs, labels = zip(*batch)

    titles = nn.utils.rnn.pad_sequence(
        titles, batch_first=True, padding_value=PAD_IDX
    )
    diffs = nn.utils.rnn.pad_sequence(
        diffs, batch_first=True, padding_value=PAD_IDX
    )

    labels = torch.stack(labels)
    return titles, diffs, labels

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["IS_REVERSED"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["IS_REVERSED"]
)

MAX_TITLE_LEN = 32
MAX_DIFF_LEN = 256
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-3

train_dataset = DiffDataset(train_df, vocab, MAX_TITLE_LEN, MAX_DIFF_LEN)
val_dataset = DiffDataset(val_df, vocab, MAX_TITLE_LEN, MAX_DIFF_LEN)
test_dataset = DiffDataset(test_df, vocab, MAX_TITLE_LEN, MAX_DIFF_LEN)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class DualTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
        pad_idx=0,
        max_len=512
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.pos_encoder = PositionalEncoding(
            embed_dim,
            dropout=dropout,
            max_len=max_len
        )

        title_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.title_encoder = nn.TransformerEncoder(
            title_layer,
            num_layers=num_layers
        )

        diff_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.diff_encoder = nn.TransformerEncoder(
            diff_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )

    def _encode(self, x, encoder):
        emb = self.embedding(x) * math.sqrt(self.embed_dim)
        emb = self.pos_encoder(emb)

        key_padding_mask = x.eq(self.pad_idx)
        out = encoder(emb, src_key_padding_mask=key_padding_mask)
        return out, key_padding_mask

    def _pool(self, encoded, key_padding_mask):
        valid_mask = (~key_padding_mask).unsqueeze(-1).type_as(encoded)
        summed = (encoded * valid_mask).sum(dim=1)
        denom = valid_mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, title, diff):
        title_out, title_mask = self._encode(title, self.title_encoder)
        diff_out, diff_mask = self._encode(diff, self.diff_encoder)

        title_vec = self._pool(title_out, title_mask)
        diff_vec = self._pool(diff_out, diff_mask)

        combined = torch.cat([title_vec, diff_vec], dim=1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)
        return logits.squeeze(1)

model = DualTransformer(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
    pad_idx=PAD_IDX,
    max_len=max(MAX_TITLE_LEN, MAX_DIFF_LEN)
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    for title, diff, label in loader:
        title = title.to(device)
        diff = diff.to(device)
        label = label.to(device)

        logits = model(title, diff)
        loss = criterion(logits, label)
        total_loss += loss.item() * label.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == label).sum().item()
        total += label.size(0)

    model.train()
    return total_loss / max(1, total), correct / max(1, total)

for epoch in range(1, EPOCHS + 1):
    train_loss = 0.0
    total = 0

    for title, diff, label in train_loader:
        title = title.to(device)
        diff = diff.to(device)
        label = label.to(device)

        logits = model(title, diff)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * label.size(0)
        total += label.size(0)

    train_loss /= max(1, total)
    val_loss, val_acc = evaluate(model, val_loader)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test: loss={test_loss:.4f} | acc={test_acc:.4f}")

save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

with open(os.path.join(save_dir, "vocab_stoi.json"), "w", encoding="utf-8") as f:
    json.dump(dict(vocab.stoi), f)

config = {
    "max_title_len": MAX_TITLE_LEN,
    "max_diff_len": MAX_DIFF_LEN,
    "max_seq_len": max(MAX_TITLE_LEN, MAX_DIFF_LEN),
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "ff_dim": FF_DIM,
    "dropout": DROPOUT
}

with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("Model exported to:", save_dir)
