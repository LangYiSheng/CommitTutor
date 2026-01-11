from detector.model import DefectDetector
from git_utils.models import CommitData
from detector.registry import register_detector
from detector.lstm_v1.train.git_diff_tools import extract_diff_payload

import os
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
import json
from collections import Counter
from torchtext.vocab import Vocab



class DualLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.lstm_title = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
        )

        self.lstm_diff = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
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

@register_detector("LSTM_v1")
class LSTMV1Detector(DefectDetector):

    def load(self):
        if self._loaded:
            return

        base_dir = os.path.dirname(__file__)

        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # tokenizer
        self.tokenizer = get_tokenizer("basic_english")

        # load config
        with open(os.path.join(base_dir, "config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # load vocab from json (SAFE)
        with open(os.path.join(base_dir, "vocab_stoi.json"), "r", encoding="utf-8") as f:
            stoi = json.load(f)

        # rebuild vocab object
        self.vocab = Vocab(Counter())
        self.vocab.stoi = stoi
        self.vocab.itos = [None] * len(stoi)

        for token, idx in stoi.items():
            self.vocab.itos[idx] = token

        self.vocab.unk_index = stoi["<unk>"]
        self.UNK_IDX = stoi["<unk>"]
        self.PAD_IDX = stoi["<pad>"]

        # build model
        self.model = DualLSTM(
            vocab_size=len(self.vocab),
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            pad_idx=self.PAD_IDX
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

    def _encode(self, text: str, max_len: int):
        tokens = self.tokenizer(text)
        ids = [
            self.vocab.stoi.get(t, self.UNK_IDX)
            for t in tokens
        ][:max_len]

        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def analyze(self, commit_info: CommitData) -> float:
        """
        使用 commit message + 每个文件 diff_text
        对每个文件计算风险概率，返回最大值
        """
        self._ensure_loaded()

        # commit-level text
        message = commit_info.message or ""

        title_ids = self._encode(
            message,
            self.config["max_title_len"]
        ).to(self.device)

        file_probs = []

        file_allowlist = ['.java']

        with torch.no_grad():
            for file in commit_info.files:
                # 只分析特定类型的文件
                if not any(file.file_path.endswith(ext) for ext in file_allowlist):
                    continue
                diff_text = extract_diff_payload(file.diff_text)
                # 空字符串不处理
                if not diff_text.strip():
                    continue
                diff_ids = self._encode(
                    diff_text,
                    self.config["max_diff_len"]
                ).to(self.device)

                prob = self.model(title_ids, diff_ids).item()
                file_probs.append(prob)

        # 如果没有文件（极端情况）
        if not file_probs:
            return 0.0

        print(f"DEBUG: file_probs={file_probs}")

        # 使用最大风险作为 commit 风险
        return float(max(file_probs))

__all__ = ["LSTMV1Detector"]
