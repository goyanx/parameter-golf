from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    attn_kv_heads: int | None = None
    n_layers: int = 2
    max_seq_len: int = 128
    dropout: float = 0.0
    weight_sharing: bool = False
    positional_encoding: str = "learned"


def _alibi_slopes(n_heads: int, device: torch.device) -> torch.Tensor:
    # Matches the ALiBi paper construction used in many reference impls.
    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2.0 ** (-2.0 ** (-(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _get_slopes_power_of_2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = _get_slopes_power_of_2(closest)
        extra = _get_slopes_power_of_2(2 * closest)[0::2]
        slopes += extra[: n_heads - closest]
    return torch.tensor(slopes, dtype=torch.float32, device=device)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_heads = n_heads if kv_heads is None else kv_heads
        if self.kv_heads <= 0 or self.kv_heads > n_heads:
            raise ValueError("attn_kv_heads must be in [1, n_heads]")
        if n_heads % self.kv_heads != 0:
            raise ValueError("n_heads must be divisible by attn_kv_heads")
        self.head_group_size = n_heads // self.kv_heads

        kv_width = self.kv_heads * self.head_dim
        self.qkv = nn.Linear(d_model, d_model + 2 * kv_width)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        alibi_bias: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        b, t, c = x.shape
        kv_width = self.kv_heads * self.head_dim
        qkv = self.qkv(x)
        q = qkv[..., : self.d_model]
        k = qkv[..., self.d_model : self.d_model + kv_width]
        v = qkv[..., self.d_model + kv_width :]

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.kv_heads != self.n_heads:
            k = k.repeat_interleave(self.head_group_size, dim=1)
            v = v.repeat_interleave(self.head_group_size, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if alibi_bias is not None:
            att = att + alibi_bias
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.proj(out)
        if return_attn:
            return out, att
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        hidden = 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        attn_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            dropout,
            kv_heads=attn_kv_heads,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        alibi_bias: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        if return_attn:
            attn_out, attn_probs = self.attn(self.ln1(x), attn_mask, alibi_bias=alibi_bias, return_attn=True)
        else:
            attn_out = self.attn(self.ln1(x), attn_mask, alibi_bias=alibi_bias)
            attn_probs = None
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        if return_attn:
            return x, attn_probs
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.positional_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        elif cfg.positional_encoding == "alibi":
            self.pos_emb = None
        else:
            raise ValueError("positional_encoding must be 'learned' or 'alibi'")
        self.drop = nn.Dropout(cfg.dropout)

        if cfg.weight_sharing:
            self.shared_block = TransformerBlock(
                cfg.d_model,
                cfg.n_heads,
                cfg.dropout,
                attn_kv_heads=cfg.attn_kv_heads,
            )
            self.blocks = None
        else:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        cfg.d_model,
                        cfg.n_heads,
                        cfg.dropout,
                        attn_kv_heads=cfg.attn_kv_heads,
                    )
                    for _ in range(cfg.n_layers)
                ]
            )
            self.shared_block = None

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def _attn_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones((1, 1, seq_len, seq_len), device=device))

    def _alibi_bias(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.cfg.positional_encoding != "alibi":
            return None
        slopes = _alibi_slopes(self.cfg.n_heads, device=device).view(1, self.cfg.n_heads, 1, 1)
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        rel = pos.view(1, 1, 1, seq_len) - pos.view(1, 1, seq_len, 1)
        rel = rel.clamp_max(0.0)
        return rel * slopes

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        return_attn: bool = False,
    ):
        b, t = idx.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Input length {t} exceeds max_seq_len={self.cfg.max_seq_len}")

        x = self.token_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        mask = self._attn_mask(t, idx.device)
        alibi_bias = self._alibi_bias(t, idx.device)
        attn_maps = [] if return_attn else None

        if self.cfg.weight_sharing:
            assert self.shared_block is not None
            for _ in range(self.cfg.n_layers):
                if return_attn:
                    x, a = self.shared_block(x, mask, alibi_bias=alibi_bias, return_attn=True)
                    assert attn_maps is not None
                    attn_maps.append(a)
                else:
                    x = self.shared_block(x, mask, alibi_bias=alibi_bias)
        else:
            assert self.blocks is not None
            for blk in self.blocks:
                if return_attn:
                    x, a = blk(x, mask, alibi_bias=alibi_bias, return_attn=True)
                    assert attn_maps is not None
                    attn_maps.append(a)
                else:
                    x = blk(x, mask, alibi_bias=alibi_bias)

        x = self.ln_f(x)
        hidden = x
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_hidden and return_attn:
            return logits, loss, hidden, attn_maps
        if return_hidden:
            return logits, loss, hidden
        if return_attn:
            return logits, loss, attn_maps
        return logits, loss
