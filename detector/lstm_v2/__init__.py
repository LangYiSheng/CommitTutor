from detector.model import DefectDetector
from git_utils.models import CommitData
from detector.registry import register_detector
from detector.constants import FILE_ALLOWLIST
from detector.lstm_v2.train.git_diff_tools import extract_diff_payload

import os
import json
from collections import Counter

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, x, mask):
        h = torch.tanh(self.proj(x))
        score = self.v(h).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, dim=1)
        ctx = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
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
        emb = self.drop(self.embedding(ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        bsz, tlen, _ = out.size()
        idx = torch.arange(tlen, device=out.device).unsqueeze(0).expand(bsz, tlen)
        mask = idx < lengths.unsqueeze(1)
        vec, _ = self.attn(out, mask)
        return vec


class HANGatedDiffEncoder(nn.Module):
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

    def forward(self, diff_ids, line_lens, word_lens):
        bsz, lines, tok = diff_ids.size()
        flat_ids = diff_ids.view(bsz * lines, tok)
        flat_wlens = word_lens.view(bsz * lines)

        emb = self.drop(self.embedding(flat_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, flat_wlens.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.word_lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        n, tlen, _ = out.size()
        idx = torch.arange(tlen, device=out.device).unsqueeze(0).expand(n, tlen)
        wmask = idx < flat_wlens.unsqueeze(1)
        line_vec, _ = self.word_attn(out, wmask)

        line_vec = line_vec.view(bsz, lines, self.word_out_dim)

        packed2 = nn.utils.rnn.pack_padded_sequence(
            line_vec, line_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out2_packed, _ = self.line_lstm(packed2)
        out2, _ = nn.utils.rnn.pad_packed_sequence(out2_packed, batch_first=True)

        bsz2, l2, _ = out2.size()
        idx2 = torch.arange(l2, device=out2.device).unsqueeze(0).expand(bsz2, l2)
        lmask = idx2 < line_lens.unsqueeze(1)
        diff_vec, _ = self.line_attn(out2, lmask)
        return diff_vec


class HANCommitClassifier(nn.Module):
    def __init__(self, vocab_size, pad_idx, config):
        super().__init__()
        self.title_enc = TitleEncoder(
            vocab_size=vocab_size,
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim_title"],
            pad_idx=pad_idx,
            dropout=config["dropout"],
            bidirectional=config["bidirectional"]
        )
        self.diff_enc = HANGatedDiffEncoder(
            vocab_size=vocab_size,
            embed_dim=config["embed_dim"],
            hidden_word=config["hidden_dim_word"],
            hidden_line=config["hidden_dim_line"],
            pad_idx=pad_idx,
            dropout=config["dropout"],
            bidirectional=config["bidirectional"]
        )

        title_dim = config["hidden_dim_title"] * (2 if config["bidirectional"] else 1)
        diff_dim = config["hidden_dim_line"] * (2 if config["bidirectional"] else 1)
        fusion_dim = title_dim + diff_dim
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, titles, diffs, title_lens, line_lens, word_lens):
        title_vec = self.title_enc(titles, title_lens)
        diff_vec = self.diff_enc(diffs, line_lens, word_lens)
        h = torch.cat([title_vec, diff_vec], dim=1)
        logits = self.fc(h).squeeze(1)
        return logits


def split_diff_lines(diff_text: str):
    s = str(diff_text)
    lines = s.splitlines()
    if len(lines) == 0:
        return [""]
    return lines


@register_detector("LSTM_v2")
class LSTMV2Detector(DefectDetector):
    def load(self):
        if self._loaded:
            return

        base_dir = os.path.dirname(__file__)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = get_tokenizer("basic_english")

        with open(os.path.join(base_dir, "config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)

        with open(os.path.join(base_dir, "vocab_stoi.json"), "r", encoding="utf-8") as f:
            stoi = json.load(f)

        self.vocab = Vocab(Counter())
        self.vocab.stoi = stoi
        self.vocab.itos = [None] * len(stoi)

        for token, idx in stoi.items():
            self.vocab.itos[idx] = token

        self.vocab.unk_index = stoi["<unk>"]
        self.UNK_IDX = stoi["<unk>"]
        self.PAD_IDX = stoi["<pad>"]

        self.model = HANCommitClassifier(
            vocab_size=len(self.vocab),
            pad_idx=self.PAD_IDX,
            config=self.config
        )

        self.model.load_state_dict(
            torch.load(
                os.path.join(base_dir, "model.pt"),
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def unload(self):
        self.model = None
        self.vocab = None
        self._loaded = False

    def _encode_tokens(self, tokens, max_len):
        if tokens is None or len(tokens) == 0:
            tokens = [""]
        ids = [self.vocab.stoi.get(t, self.UNK_IDX) for t in tokens][:max_len]
        if len(ids) == 0:
            ids = [self.UNK_IDX]
        return torch.tensor(ids, dtype=torch.long)

    def _encode_title(self, text):
        toks = self.tokenizer(str(text))
        return self._encode_tokens(toks, self.config["max_title_len"])

    def _encode_diff_han(self, diff_text):
        lines = split_diff_lines(diff_text)[: self.config["max_diff_lines"]]
        if len(lines) == 0:
            lines = [""]

        line_tensors = []
        for ln in lines:
            toks = self.tokenizer(ln)
            line_tensors.append(self._encode_tokens(toks, self.config["max_tokens_per_line"]))

        if len(line_tensors) == 0:
            line_tensors = [torch.tensor([self.UNK_IDX], dtype=torch.long)]

        return line_tensors

    def _build_diff_tensors(self, diff_lines):
        max_lines = min(self.config["max_diff_lines"], max(1, len(diff_lines)))
        max_tok = 1
        for ln in diff_lines[:max_lines]:
            max_tok = max(max_tok, int(ln.numel()))
        max_tok = min(self.config["max_tokens_per_line"], max_tok)
        if max_tok < 1:
            max_tok = 1

        diffs_pad = torch.full((1, max_lines, max_tok), self.PAD_IDX, dtype=torch.long)
        word_lens = torch.ones((1, max_lines), dtype=torch.long)
        line_lens = torch.zeros((1,), dtype=torch.long)

        n_lines = min(max_lines, len(diff_lines))
        if n_lines < 1:
            n_lines = 1
            diff_lines = [torch.tensor([self.PAD_IDX], dtype=torch.long)]

        line_lens[0] = n_lines

        for j in range(max_lines):
            if j < len(diff_lines):
                ln = diff_lines[j][:max_tok]
                diffs_pad[0, j, :ln.numel()] = ln
                word_lens[0, j] = max(1, ln.numel())
            else:
                word_lens[0, j] = 1

        return diffs_pad, line_lens, word_lens

    def analyze(self, commit_info: CommitData):
        self._ensure_loaded()

        message = commit_info.message or ""
        title_ids = self._encode_title(message)
        title_lens = torch.tensor([max(1, title_ids.numel())], dtype=torch.long)
        title_ids = title_ids.unsqueeze(0)

        file_probs = []
        with torch.no_grad():
            for file in commit_info.files:
                if not any(file.file_path.endswith(ext) for ext in FILE_ALLOWLIST):
                    continue
                diff_text = extract_diff_payload(file.diff_text)
                if not diff_text.strip():
                    continue

                diff_lines = self._encode_diff_han(diff_text)
                diffs_pad, line_lens, word_lens = self._build_diff_tensors(diff_lines)

                logits = self.model(
                    title_ids.to(self.device),
                    diffs_pad.to(self.device),
                    title_lens.to(self.device),
                    line_lens.to(self.device),
                    word_lens.to(self.device)
                )
                prob = torch.sigmoid(logits).item()
                file_probs.append(prob)

        if not file_probs:
            return 0.0

        print(f"DEBUG: file_probs={file_probs}")
        return float(max(file_probs))


__all__ = ["LSTMV2Detector"]
