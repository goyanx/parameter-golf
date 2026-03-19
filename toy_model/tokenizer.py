from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ByteTokenizer:
    pad_token_id: int = 0

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, tokens: List[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def train_val_tokens(text: str, val_frac: float = 0.1) -> Dict[str, List[int]]:
    tok = ByteTokenizer()
    data = tok.encode(text)
    split = int((1.0 - val_frac) * len(data))
    split = max(1, min(split, len(data) - 1))
    return {"train": data[:split], "val": data[split:]}

