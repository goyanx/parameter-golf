from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def magnitude_prune_model(model: nn.Module, amount: float = 0.1) -> Dict[str, float]:
    if amount < 0.0 or amount >= 1.0:
        raise ValueError("amount must be in [0.0, 1.0)")
    stats: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or param.ndim < 2:
            continue
        with torch.no_grad():
            flat = param.detach().abs().reshape(-1)
            if flat.numel() == 0:
                continue
            k = int(amount * flat.numel())
            if k <= 0:
                stats[name] = 0.0
                continue
            threshold = torch.kthvalue(flat, k).values
            mask = param.detach().abs() > threshold
            param.mul_(mask)
            sparsity = 1.0 - mask.float().mean().item()
            stats[name] = sparsity
    return stats


def global_sparsity(model: nn.Module) -> Tuple[int, int, float]:
    total = 0
    nonzero = 0
    with torch.no_grad():
        for p in model.parameters():
            total += p.numel()
            nonzero += int((p != 0).sum().item())
    sparsity = 1.0 - (nonzero / total if total else 0.0)
    return nonzero, total, sparsity

