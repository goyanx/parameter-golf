from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


@dataclass
class QuantizedTensor:
    qweight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    bits: int
    packed_bytes: bytes | None = None


@dataclass
class QuantConfig:
    bits: int = 4
    group_size: int = 64
    exclude_patterns: tuple[str, ...] = ("token_emb", "head", "ln", "norm")
    fallback_dtype: str = "fp16"
    pack_order: str = "state_dict"
    layer_bits: tuple[tuple[str, int], ...] = ()


def _quant_bounds(bits: int) -> Tuple[int, int]:
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8]")
    qmin = 0
    qmax = (1 << bits) - 1
    return qmin, qmax


def quantize_tensor_affine(t: torch.Tensor, bits: int = 8) -> QuantizedTensor:
    qmin, qmax = _quant_bounds(bits)
    x_min = t.min()
    x_max = t.max()

    # Guard against zero range tensors.
    if float(x_max - x_min) < 1e-8:
        scale = torch.tensor(1.0, device=t.device, dtype=torch.float32)
        zero_point = torch.tensor(0.0, device=t.device, dtype=torch.float32)
        q = torch.zeros_like(t, dtype=torch.uint8)
        return QuantizedTensor(qweight=q, scale=scale, zero_point=zero_point, bits=bits)

    scale = (x_max - x_min) / float(qmax - qmin)
    zero_point = qmin - torch.round(x_min / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)

    q = torch.round(t / scale + zero_point)
    q = torch.clamp(q, qmin, qmax).to(torch.uint8)
    return QuantizedTensor(qweight=q, scale=scale.float(), zero_point=zero_point.float(), bits=bits)


def dequantize_tensor_affine(qt: QuantizedTensor) -> torch.Tensor:
    return (qt.qweight.float() - qt.zero_point) * qt.scale


def _should_exclude(name: str, patterns: Iterable[str]) -> bool:
    n = name.lower()
    return any(p.lower() in n for p in patterns)


def _pack_nibbles_u4(q: torch.Tensor) -> bytes:
    flat = q.view(-1).to(torch.uint8)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] & 0x0F) << 4
    packed = (lo | hi).contiguous()
    return packed.numpy().tobytes()


def _pack_bits(q: torch.Tensor, bits: int) -> bytes:
    if bits == 4:
        return _pack_nibbles_u4(q)
    if bits >= 8:
        return q.view(-1).to(torch.uint8).numpy().tobytes()
    flat = q.view(-1).to(torch.uint8)
    out = bytearray()
    acc = 0
    acc_bits = 0
    mask = (1 << bits) - 1
    for v in flat.tolist():
        acc |= (int(v) & mask) << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits > 0:
        out.append(acc & 0xFF)
    return bytes(out)


def _parse_layer_bits(layer_bits: object) -> tuple[tuple[str, int], ...]:
    if not layer_bits:
        return ()
    if isinstance(layer_bits, dict):
        items = layer_bits.items()
    elif isinstance(layer_bits, list):
        parsed = []
        for entry in layer_bits:
            if not isinstance(entry, dict):
                continue
            pat = str(entry.get("pattern", "")).strip()
            bits = int(entry.get("bits", 0))
            if not pat:
                continue
            parsed.append((pat, bits))
        return tuple(parsed)
    else:
        return ()
    out = []
    for pat, bits in items:
        try:
            b = int(bits)
        except (TypeError, ValueError):
            continue
        out.append((str(pat), b))
    return tuple(out)


def _bits_for_name(name: str, default_bits: int, layer_bits: tuple[tuple[str, int], ...]) -> int:
    n = name.lower()
    bits = int(default_bits)
    for pattern, b in layer_bits:
        if pattern.lower() in n:
            bits = int(b)
    return bits


def _quantize_row_groupwise_2d(t: torch.Tensor, bits: int, group_size: int) -> QuantizedTensor:
    if t.ndim != 2:
        raise ValueError("Expected 2D tensor for row-groupwise quantization")
    qmin, qmax = _quant_bounds(bits)
    rows, cols = t.shape
    groups = (cols + group_size - 1) // group_size

    q = torch.empty_like(t, dtype=torch.uint8)
    scale = torch.empty((rows, groups), dtype=torch.float32)
    zero_point = torch.empty((rows, groups), dtype=torch.float32)

    for r in range(rows):
        row = t[r]
        for g in range(groups):
            s = g * group_size
            e = min((g + 1) * group_size, cols)
            chunk = row[s:e]
            x_min = chunk.min()
            x_max = chunk.max()
            if float(x_max - x_min) < 1e-8:
                sc = torch.tensor(1.0, dtype=torch.float32)
                zp = torch.tensor(0.0, dtype=torch.float32)
                q_chunk = torch.zeros_like(chunk, dtype=torch.uint8)
            else:
                sc = ((x_max - x_min) / float(qmax - qmin)).float()
                zp = (qmin - torch.round(x_min / sc)).clamp(qmin, qmax).float()
                q_chunk = torch.round(chunk / sc + zp).clamp(qmin, qmax).to(torch.uint8)
            q[r, s:e] = q_chunk
            scale[r, g] = sc
            zero_point[r, g] = zp

    packed = _pack_bits(q, bits=bits) if bits < 8 else None
    return QuantizedTensor(qweight=q, scale=scale, zero_point=zero_point, bits=bits, packed_bytes=packed)


def quantize_tensor_groupwise(t: torch.Tensor, bits: int = 4, group_size: int = 64) -> QuantizedTensor:
    if t.ndim < 2:
        return quantize_tensor_affine(t, bits=bits)
    original_shape = t.shape
    t2 = t.reshape(t.shape[0], -1)
    qt = _quantize_row_groupwise_2d(t2, bits=bits, group_size=group_size)
    qt.qweight = qt.qweight.reshape(original_shape)
    return qt


def dequantize_tensor_groupwise(qt: QuantizedTensor, original_shape: torch.Size, group_size: int = 64) -> torch.Tensor:
    if len(original_shape) < 2:
        return dequantize_tensor_affine(qt).reshape(original_shape)
    q2 = qt.qweight.reshape(original_shape[0], -1).float()
    rows, cols = q2.shape
    groups = (cols + group_size - 1) // group_size
    out = torch.empty_like(q2)
    for r in range(rows):
        for g in range(groups):
            s = g * group_size
            e = min((g + 1) * group_size, cols)
            sc = qt.scale[r, g]
            zp = qt.zero_point[r, g]
            out[r, s:e] = (q2[r, s:e] - zp) * sc
    return out.reshape(original_shape)


def quantize_model_state_dict(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 64,
    exclude_patterns: Iterable[str] = (),
) -> Dict[str, QuantizedTensor]:
    qstate: Dict[str, QuantizedTensor] = {}
    for name, param in model.state_dict().items():
        if not torch.is_floating_point(param):
            continue
        p = param.detach().cpu()
        if _should_exclude(name, exclude_patterns):
            qstate[name] = QuantizedTensor(
                qweight=p.half().view(torch.uint8),
                scale=torch.tensor(1.0),
                zero_point=torch.tensor(0.0),
                bits=16,
            )
            continue
        if bits == 4:
            qstate[name] = quantize_tensor_groupwise(p, bits=bits, group_size=group_size)
        else:
            qstate[name] = quantize_tensor_affine(p, bits=bits)
    return qstate


def fake_quantize_model_inplace(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
    exclude_patterns: Iterable[str] = ("token_emb", "head", "ln", "norm"),
    layer_bits: object = None,
) -> None:
    overrides = _parse_layer_bits(layer_bits)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad or not torch.is_floating_point(p):
                continue
            if _should_exclude(name, exclude_patterns):
                continue
            b = _bits_for_name(name, default_bits=int(bits), layer_bits=overrides)
            cpu = p.detach().cpu()
            if b <= 7 and cpu.ndim >= 2:
                qt = quantize_tensor_groupwise(cpu, bits=b, group_size=group_size)
                dq = dequantize_tensor_groupwise(qt, cpu.shape, group_size=group_size)
            else:
                qt = quantize_tensor_affine(cpu, bits=b)
                dq = dequantize_tensor_affine(qt).reshape(cpu.shape)
            p.copy_(dq.to(device=p.device, dtype=p.dtype))


def quantized_payload_bytes(model: nn.Module, cfg: QuantConfig) -> bytes:
    chunks = []
    layer_bits = _parse_layer_bits(cfg.layer_bits)
    items = list(model.state_dict().items())
    if cfg.pack_order == "name":
        items = sorted(items, key=lambda kv: kv[0])
    elif cfg.pack_order == "size_desc":
        items = sorted(items, key=lambda kv: (-kv[1].numel(), kv[0]))
    elif cfg.pack_order != "state_dict":
        raise ValueError("pack_order must be 'state_dict', 'name', or 'size_desc'")

    for name, p in items:
        if not torch.is_floating_point(p):
            continue
        t = p.detach().cpu()
        if _should_exclude(name, cfg.exclude_patterns):
            if cfg.fallback_dtype == "fp32":
                payload = t.float().numpy().tobytes()
            else:
                payload = t.half().numpy().tobytes()
            chunks.append(payload)
            continue
        b = _bits_for_name(name, default_bits=int(cfg.bits), layer_bits=layer_bits)
        if b <= 7 and t.ndim >= 2:
            qt = quantize_tensor_groupwise(t, bits=b, group_size=cfg.group_size)
            chunks.append(qt.packed_bytes or qt.qweight.numpy().tobytes())
            chunks.append(qt.scale.numpy().tobytes())
            chunks.append(qt.zero_point.numpy().tobytes())
        else:
            qt = quantize_tensor_affine(t, bits=b)
            chunks.append(qt.qweight.numpy().tobytes())
            chunks.append(qt.scale.numpy().tobytes())
            chunks.append(qt.zero_point.numpy().tobytes())
    raw = b"".join(chunks)
    return zlib.compress(raw, level=9)
