import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv("final_diff_dataset.csv")
df = df[["PR_TITLE", "DIFF_CODE", "IS_REVERSED"]].dropna()

tokenizer = get_tokenizer("basic_english")

counter = Counter()

for text in df["PR_TITLE"]:
    counter.update(tokenizer(str(text)))

for text in df["DIFF_CODE"]:
    counter.update(tokenizer(str(text)))

vocab = Vocab(
    counter,
    specials=["<pad>", "<unk>"]
)

vocab.unk_index = vocab.stoi["<unk>"]
PAD_IDX = vocab.stoi["<pad>"]

print("Vocab size:", len(vocab))

class DiffDataset(Dataset):
    def __init__(self, df, vocab, max_title_len=20, max_diff_len=100):
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

train_dataset = DiffDataset(train_df, vocab)
val_dataset   = DiffDataset(val_df, vocab)
test_dataset  = DiffDataset(test_df, vocab)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

class DualLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=PAD_IDX
        )

        self.lstm_title = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, dropout=0.3
        )
        self.lstm_diff = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, dropout=0.3
        )

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, title, diff):
        title_emb = self.embedding(title)
        diff_emb  = self.embedding(diff)

        _, (h_title, _) = self.lstm_title(title_emb)
        _, (h_diff, _)  = self.lstm_diff(diff_emb)

        h = torch.cat([h_title[-1], h_diff[-1]], dim=1)
        out = self.fc(h)

        return torch.sigmoid(out).squeeze(1)

model = DualLSTM(len(vocab)).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for title, diff, label in loader:
            title = title.to(device)
            diff  = diff.to(device)
            label = label.to(device)

            pred = model(title, diff)
            loss = criterion(pred, label)

            total_loss += loss.item() * label.size(0)
            total += label.size(0)

            preds = (pred >= 0.5).float()
            correct += (preds == label).sum().item()

    model.train()
    return total_loss / max(1, total), correct / max(1, total)

EPOCHS = 3

for epoch in range(EPOCHS):
    train_loss = 0.0
    total = 0

    for title, diff, label in train_loader:
        title = title.to(device)
        diff  = diff.to(device)
        label = label.to(device)

        pred = model(title, diff)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * label.size(0)
        total += label.size(0)

    train_loss = train_loss / max(1, total)
    val_loss, val_acc = evaluate(model, val_loader)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test: loss={test_loss:.4f} | acc={test_acc:.4f}")

import os
import json

save_dir = ".."
os.makedirs(save_dir, exist_ok=True)

# 1. 保存模型参数
torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

# 2. 保存 vocab
import json

with open(os.path.join(save_dir,"vocab_stoi.json"), "w", encoding="utf-8") as f:
    json.dump(dict(vocab.stoi), f)

# 3. 保存配置（给 CLI 用）
config = {
    "max_title_len": 20,
    "max_diff_len": 100,
    "embed_dim": 128,
    "hidden_dim": 128
}

with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("Model exported to:", save_dir)
