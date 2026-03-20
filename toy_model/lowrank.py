from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError("rank must be in (0, min(in_features, out_features)]")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.a = nn.Parameter(torch.empty(out_features, rank))
        self.b = nn.Parameter(torch.empty(rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.b, a=math.sqrt(5))
        nn.init.zeros_(self.a)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W ~= A @ B with A[out, rank] and B[rank, in]
        w = self.a @ self.b
        return torch.nn.functional.linear(x, w, self.bias)


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    n = name.lower()
    return any(p.lower() in n for p in patterns)


def maybe_replace_linear(
    module: nn.Module,
    rank: int | None = None,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
    _prefix: str = "",
) -> nn.Module:
    if rank is None:
        return module
    if isinstance(module, nn.Linear):
        if exclude_patterns and _matches_any(_prefix, exclude_patterns):
            return module
        if include_patterns and not _matches_any(_prefix, include_patterns):
            return module
        return LowRankLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            bias=module.bias is not None,
        )
    for name, child in list(module.named_children()):
        child_prefix = f"{_prefix}.{name}" if _prefix else name
        setattr(
            module,
            name,
            maybe_replace_linear(
                child,
                rank=rank,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                _prefix=child_prefix,
            ),
        )
    return module
