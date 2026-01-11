from detector.model import DefectDetector
from git_utils.models import CommitData
from detector.registry import register_detector
from detector.constants import FILE_ALLOWLIST
from detector.lstm_v2.train.git_diff_tools import extract_diff_payload

import json
import math
import os
import warnings
from collections import Counter

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

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
        embed_dim,
        num_heads,
        num_layers,
        ff_dim,
        dropout,
        pad_idx,
        max_len
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

@register_detector("TRANSFORMER_V1")
class TransformerV1Detector(DefectDetector):
    def load(self):
        if self._loaded:
            return

        base_dir = os.path.dirname(__file__)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                "The PyTorch API of nested tensors is in prototype stage and will change "
                "in the near future\\."
            ),
            category=UserWarning
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

        max_seq_len = self.config.get(
            "max_seq_len",
            max(self.config["max_title_len"], self.config["max_diff_len"])
        )

        self.model = DualTransformer(
            vocab_size=len(self.vocab),
            embed_dim=self.config["embed_dim"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            ff_dim=self.config["ff_dim"],
            dropout=self.config["dropout"],
            pad_idx=self.PAD_IDX,
            max_len=max_seq_len
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

    def _encode(self, text, max_len):
        tokens = self.tokenizer(str(text))
        if not tokens:
            tokens = [""]
        ids = [
            self.vocab.stoi.get(t, self.UNK_IDX)
            for t in tokens
        ][:max_len]
        if not ids:
            ids = [self.UNK_IDX]

        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def analyze(self, commit_info: CommitData) -> float:
        self._ensure_loaded()

        message = commit_info.message or ""
        title_ids = self._encode(
            message,
            self.config["max_title_len"]
        ).to(self.device)

        file_probs = []
        with torch.no_grad():
            for file in commit_info.files:
                if not any(file.file_path.endswith(ext) for ext in FILE_ALLOWLIST):
                    continue

                diff_text = extract_diff_payload(file.diff_text)
                if not diff_text.strip():
                    continue

                diff_ids = self._encode(
                    diff_text,
                    self.config["max_diff_len"]
                ).to(self.device)

                logits = self.model(title_ids, diff_ids)
                prob = torch.sigmoid(logits).item()
                file_probs.append(prob)

        if not file_probs:
            return 0.0

        print(f"DEBUG: file_probs={file_probs}")
        return float(max(file_probs))

__all__ = ["TransformerV1Detector"]
