from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    n = name.lower()
    return any(p.lower() in n for p in patterns)


def _include_parameter(name: str, include_patterns: tuple[str, ...], exclude_patterns: tuple[str, ...]) -> bool:
    if exclude_patterns and _matches_any(name, exclude_patterns):
        return False
    if include_patterns and not _matches_any(name, include_patterns):
        return False
    return True


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


def structured_prune_model(
    model: nn.Module,
    amount: float = 0.1,
    mode: str = "row",
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
) -> Dict[str, float]:
    if amount < 0.0 or amount >= 1.0:
        raise ValueError("amount must be in [0.0, 1.0)")
    if mode not in {"row", "col"}:
        raise ValueError("mode must be 'row' or 'col'")

    stats: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or param.ndim < 2:
            continue
        if not _include_parameter(name, include_patterns, exclude_patterns):
            continue

        with torch.no_grad():
            view = param.detach().reshape(param.shape[0], -1)
            if mode == "col":
                view = view.t().contiguous()

            units = view.shape[0]
            k = int(amount * units)
            if k <= 0:
                stats[name] = 0.0
                continue

            scores = view.abs().mean(dim=1)
            prune_idx = torch.topk(scores, k, largest=False).indices
            mask = torch.ones_like(view)
            mask[prune_idx] = 0
            if mode == "col":
                mask = mask.t().contiguous()
            param.mul_(mask.reshape_as(param))
            stats[name] = 1.0 - float(mask.float().mean().item())
    return stats


def nm_2_4_prune_model(
    model: nn.Module,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or param.ndim < 2:
            continue
        if not _include_parameter(name, include_patterns, exclude_patterns):
            continue
        with torch.no_grad():
            view = param.detach().reshape(param.shape[0], -1)
            rows, cols = view.shape
            cols4 = (cols // 4) * 4
            if cols4 <= 0:
                stats[name] = 0.0
                continue
            core = view[:, :cols4].reshape(rows, cols4 // 4, 4)
            keep_idx = torch.topk(core.abs(), k=2, dim=2, largest=True).indices
            mask = torch.zeros_like(core)
            mask.scatter_(2, keep_idx, 1.0)
            core.mul_(mask)
            view[:, :cols4] = core.reshape(rows, cols4)
            param.copy_(view.reshape_as(param))
            stats[name] = 1.0 - float((view != 0).float().mean().item())
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
