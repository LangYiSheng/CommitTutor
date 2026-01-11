import os
import json
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
from sklearn.model_selection import train_test_split

CSV_PATH = "final_diff_dataset.csv"

MAX_TITLE_LEN = 24          # 标题最大 token 数
MAX_DIFF_LINES = 80         # diff 最大行数（HAN 的“句子数”）
MAX_TOKENS_PER_LINE = 20    # 每行最大 token 数（HAN 的“词数”）

EMBED_DIM = 128
HIDDEN_DIM_WORD = 128       # 行级（word-level）LSTM hidden
HIDDEN_DIM_LINE = 128       # diff级（line-level）LSTM hidden
HIDDEN_DIM_TITLE = 128      # title encoder hidden

BIDIRECTIONAL = True        # 建议 True
DROPOUT = 0.3

BATCH_SIZE = 32
EPOCHS = 2
LR = 1e-3
SEED = 42

SAVE_DIR = ".."  # 导出到上级目录

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = get_tokenizer("basic_english")

def split_diff_lines(diff_text: str):
    """把 diff 文本按行切分；至少返回 1 行，避免空输入。"""
    s = str(diff_text)
    lines = s.splitlines()
    if len(lines) == 0:
        return [""]  # 至少一行
    return lines

df = pd.read_csv(CSV_PATH)
df = df[["PR_TITLE", "DIFF_CODE", "IS_REVERSED"]].dropna().reset_index(drop=True)

counter = Counter()

# title tokens
for text in df["PR_TITLE"]:
    toks = tokenizer(str(text))
    counter.update(toks)

# diff tokens (按行分割再分词：更贴近 HAN 的输入形态)
for diff in df["DIFF_CODE"]:
    for line in split_diff_lines(diff):
        toks = tokenizer(line)
        counter.update(toks)

vocab = Vocab(counter, specials=["<pad>", "<unk>"])
vocab.unk_index = vocab.stoi["<unk>"]
PAD_IDX = vocab.stoi["<pad>"]

print("Vocab size:", len(vocab))

class CommitHANDataset(Dataset):
    """
    返回：
      - title_ids: [T_title]
      - diff_ids:  [L_lines, T_line]
      - label: float(0/1)
    """
    def __init__(self, df_, vocab_):
        self.df = df_.reset_index(drop=True)
        self.vocab = vocab_

    def encode_tokens(self, tokens, max_len):
        if tokens is None or len(tokens) == 0:
            tokens = [""]  # 至少一个 token（会变成 <unk> 或 pad）
        ids = [self.vocab.stoi.get(t, self.vocab.unk_index) for t in tokens][:max_len]
        if len(ids) == 0:
            ids = [self.vocab.unk_index]
        return torch.tensor(ids, dtype=torch.long)

    def encode_title(self, title_text):
        toks = tokenizer(str(title_text))
        return self.encode_tokens(toks, MAX_TITLE_LEN)

    def encode_diff_han(self, diff_text):
        lines = split_diff_lines(diff_text)[:MAX_DIFF_LINES]
        if len(lines) == 0:
            lines = [""]

        line_tensors = []
        for ln in lines:
            toks = tokenizer(ln)
            line_tensors.append(self.encode_tokens(toks, MAX_TOKENS_PER_LINE))

        # 至少一行
        if len(line_tensors) == 0:
            line_tensors = [torch.tensor([self.vocab.unk_index], dtype=torch.long)]

        return line_tensors  # list[tensor([t])], length = num_lines

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        title_ids = self.encode_title(row["PR_TITLE"])
        diff_lines = self.encode_diff_han(row["DIFF_CODE"])  # list of tensors (variable length)
        label = torch.tensor(float(row["IS_REVERSED"]), dtype=torch.float)
        return title_ids, diff_lines, label

    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    """
    输出：
      titles: [B, Tt]
      diffs:  [B, L, Tl]
      title_lens: [B]
      line_lens:  [B]            每个样本的有效行数
      word_lens:  [B, L]         每行有效 token 数（>=1）
      labels: [B]
    """
    titles, difflines_list, labels = zip(*batch)

    # ---- titles pad
    title_lens = torch.tensor([max(1, t.numel()) for t in titles], dtype=torch.long)
    titles_pad = nn.utils.rnn.pad_sequence(titles, batch_first=True, padding_value=PAD_IDX)  # [B, Tt]

    # ---- diffs pad: first decide max lines and max tokens per line within batch (cap by global max)
    B = len(difflines_list)
    max_lines = min(MAX_DIFF_LINES, max(len(x) for x in difflines_list))
    if max_lines < 1:
        max_lines = 1

    # find max tokens per line inside batch (cap by global)
    max_tok = 1
    for lines in difflines_list:
        for ln in lines[:max_lines]:
            max_tok = max(max_tok, int(ln.numel()))
    max_tok = min(MAX_TOKENS_PER_LINE, max_tok)
    if max_tok < 1:
        max_tok = 1

    # allocate diff tensor
    diffs_pad = torch.full((B, max_lines, max_tok), PAD_IDX, dtype=torch.long)
    word_lens = torch.ones((B, max_lines), dtype=torch.long)  # 每行至少 1，便于 pack（空行也给1）
    line_lens = torch.zeros((B,), dtype=torch.long)

    for i, lines in enumerate(difflines_list):
        lines = lines[:max_lines]
        n_lines = len(lines)
        if n_lines < 1:
            n_lines = 1
            lines = [torch.tensor([PAD_IDX], dtype=torch.long)]

        line_lens[i] = n_lines

        for j in range(max_lines):
            if j < len(lines):
                ln = lines[j]
                # truncate/pad to max_tok
                ln = ln[:max_tok]
                diffs_pad[i, j, :ln.numel()] = ln
                word_lens[i, j] = max(1, ln.numel())
            else:
                # padded line
                word_lens[i, j] = 1  # 仍保持 >=1

    labels = torch.stack(labels)  # [B]
    return titles_pad, diffs_pad, title_lens, line_lens, word_lens, labels

class AdditiveAttention(nn.Module):
    """
    输入:
      x: [N, T, D]
      mask: [N, T]  bool(True=valid, False=pad)
    输出:
      ctx: [N, D]
      alpha: [N, T]
    """
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, x, mask):
        # x: [N, T, D]
        h = torch.tanh(self.proj(x))          # [N, T, D]
        score = self.v(h).squeeze(-1)         # [N, T]

        # mask: True for valid
        score = score.masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, dim=1)   # [N, T]
        ctx = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)  # [N, D]
        return ctx, alpha

class TitleEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)

        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn = AdditiveAttention(out_dim)

    def forward(self, ids, lengths):
        # ids: [B, T]
        emb = self.drop(self.embedding(ids))  # [B, T, E]

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # [B, T, D]

        # mask
        B, T, _ = out.size()
        idx = torch.arange(T, device=out.device).unsqueeze(0).expand(B, T)
        mask = idx < lengths.unsqueeze(1)  # [B, T]

        vec, _ = self.attn(out, mask)
        return vec  # [B, D]

class HANGatedDiffEncoder(nn.Module):
    """
    HAN:
      word-level: 每行 tokens -> word BiLSTM -> attention -> line vector
      line-level: line vectors -> line BiLSTM -> attention -> diff vector
    """
    def __init__(self, vocab_size, embed_dim, hidden_word, hidden_line, pad_idx, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)

        self.bidirectional = bidirectional
        self.word_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_word,
            batch_first=True,
            bidirectional=bidirectional
        )
        word_out_dim = hidden_word * (2 if bidirectional else 1)
        self.word_attn = AdditiveAttention(word_out_dim)

        self.line_lstm = nn.LSTM(
            input_size=word_out_dim,
            hidden_size=hidden_line,
            batch_first=True,
            bidirectional=bidirectional
        )
        line_out_dim = hidden_line * (2 if bidirectional else 1)
        self.line_attn = AdditiveAttention(line_out_dim)

        self.word_out_dim = word_out_dim
        self.line_out_dim = line_out_dim

    def forward(self, diff_ids, line_lens, word_lens):
        """
        diff_ids: [B, L, Tl]
        line_lens: [B]      有效行数（>=1）
        word_lens: [B, L]   每行有效 token 数（>=1）
        """
        B, L, Tl = diff_ids.size()

        # flatten lines => N = B*L
        flat_ids = diff_ids.view(B * L, Tl)  # [N, Tl]
        flat_wlens = word_lens.view(B * L)   # [N]

        emb = self.drop(self.embedding(flat_ids))  # [N, Tl, E]

        # word-level pack
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, flat_wlens.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.word_lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # [N, Tw, Dw]

        N, Tw, _ = out.size()
        idx = torch.arange(Tw, device=out.device).unsqueeze(0).expand(N, Tw)
        wmask = idx < flat_wlens.unsqueeze(1)  # [N, Tw]
        line_vec, _ = self.word_attn(out, wmask)  # [N, Dw]

        # reshape back to [B, L, Dw]
        line_vec = line_vec.view(B, L, self.word_out_dim)

        # line-level pack
        packed2 = nn.utils.rnn.pack_padded_sequence(
            line_vec, line_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out2_packed, _ = self.line_lstm(packed2)
        out2, _ = nn.utils.rnn.pad_packed_sequence(out2_packed, batch_first=True)  # [B, L2, Dl]

        B2, L2, _ = out2.size()
        idx2 = torch.arange(L2, device=out2.device).unsqueeze(0).expand(B2, L2)
        lmask = idx2 < line_lens.unsqueeze(1)  # [B, L2]
        diff_vec, _ = self.line_attn(out2, lmask)  # [B, Dl]

        return diff_vec  # [B, Dl]

class HANCommitClassifier(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()

        # Title encoder & Diff encoder 各自有 embedding（也可以共享，但这里保持结构清晰）
        self.title_enc = TitleEncoder(
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM_TITLE,
            pad_idx=pad_idx,
            dropout=DROPOUT,
            bidirectional=BIDIRECTIONAL
        )

        self.diff_enc = HANGatedDiffEncoder(
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            hidden_word=HIDDEN_DIM_WORD,
            hidden_line=HIDDEN_DIM_LINE,
            pad_idx=pad_idx,
            dropout=DROPOUT,
            bidirectional=BIDIRECTIONAL
        )

        title_dim = HIDDEN_DIM_TITLE * (2 if BIDIRECTIONAL else 1)
        diff_dim = HIDDEN_DIM_LINE * (2 if BIDIRECTIONAL else 1)

        fusion_dim = title_dim + diff_dim
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, titles, diffs, title_lens, line_lens, word_lens):
        title_vec = self.title_enc(titles, title_lens)                 # [B, Dt]
        diff_vec = self.diff_enc(diffs, line_lens, word_lens)          # [B, Dd]
        h = torch.cat([title_vec, diff_vec], dim=1)                    # [B, Dt+Dd]
        logits = self.fc(h).squeeze(1)                                 # [B]
        return logits

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["IS_REVERSED"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["IS_REVERSED"]
)

train_ds = CommitHANDataset(train_df, vocab)
val_ds   = CommitHANDataset(val_df, vocab)
test_ds  = CommitHANDataset(test_df, vocab)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


model = HANCommitClassifier(len(vocab), PAD_IDX).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def batch_to_device(batch):
    titles, diffs, title_lens, line_lens, word_lens, labels = batch
    return (
        titles.to(device),
        diffs.to(device),
        title_lens.to(device),
        line_lens.to(device),
        word_lens.to(device),
        labels.to(device)
    )

@torch.no_grad()
def evaluate(model_, loader_):
    model_.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader_:
        titles, diffs, title_lens, line_lens, word_lens, labels = batch_to_device(batch)
        logits = model_(titles, diffs, title_lens, line_lens, word_lens)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(1, total), correct / max(1, total)

def train_one_epoch(model_, loader_):
    model_.train()
    total_loss = 0.0
    total = 0

    for batch in loader_:
        titles, diffs, title_lens, line_lens, word_lens, labels = batch_to_device(batch)

        logits = model_(titles, diffs, title_lens, line_lens, word_lens)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        # LSTM 常用：梯度裁剪，稳一点
        nn.utils.clip_grad_norm_(model_.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

    return total_loss / max(1, total)

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test: loss={test_loss:.4f} | acc={test_acc:.4f}")



os.makedirs(SAVE_DIR, exist_ok=True)

torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))

with open(os.path.join(SAVE_DIR, "vocab_stoi.json"), "w", encoding="utf-8") as f:
    json.dump(dict(vocab.stoi), f, ensure_ascii=False)

config = {
    "max_title_len": MAX_TITLE_LEN,
    "max_diff_lines": MAX_DIFF_LINES,
    "max_tokens_per_line": MAX_TOKENS_PER_LINE,
    "embed_dim": EMBED_DIM,
    "hidden_dim_word": HIDDEN_DIM_WORD,
    "hidden_dim_line": HIDDEN_DIM_LINE,
    "hidden_dim_title": HIDDEN_DIM_TITLE,
    "bidirectional": BIDIRECTIONAL,
    "dropout": DROPOUT
}
with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("Model exported to:", SAVE_DIR)
