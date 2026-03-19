from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from quantize import QuantConfig, quantized_payload_bytes


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _payload_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def _payload_from_quantized_state(model: nn.Module, quant_cfg: QuantConfig) -> bytes:
    payload = quantized_payload_bytes(model, quant_cfg)
    return payload


def estimate_model_bytes(
    model: nn.Module,
    quant_bits: int | None = None,
    quant_group_size: int = 64,
    quant_exclude_patterns: tuple[str, ...] = ("token_emb", "head", "ln", "norm"),
    quant_fallback_dtype: str = "fp16",
) -> int:
    payload = model_payload_bytes(
        model,
        quant_bits=quant_bits,
        quant_group_size=quant_group_size,
        quant_exclude_patterns=quant_exclude_patterns,
        quant_fallback_dtype=quant_fallback_dtype,
    )
    return len(payload)


def model_payload_bytes(
    model: nn.Module,
    quant_bits: int | None = None,
    quant_group_size: int = 64,
    quant_exclude_patterns: tuple[str, ...] = ("token_emb", "head", "ln", "norm"),
    quant_fallback_dtype: str = "fp16",
) -> bytes:
    if quant_bits is None:
        return _payload_from_state_dict(model.state_dict())
    qcfg = QuantConfig(
        bits=quant_bits,
        group_size=quant_group_size,
        exclude_patterns=quant_exclude_patterns,
        fallback_dtype=quant_fallback_dtype,
    )
    return _payload_from_quantized_state(model, quant_cfg=qcfg)


def estimate_artifact_bytes(
    model: nn.Module,
    code_dir: str | Path,
    quant_bits: int | None = None,
    quant_group_size: int = 64,
    quant_exclude_patterns: tuple[str, ...] = ("token_emb", "head", "ln", "norm"),
    quant_fallback_dtype: str = "fp16",
    include_files: list[str] | None = None,
) -> int:
    code_dir = Path(code_dir)
    include = include_files or [
        "model.py",
        "train.py",
        "quantize.py",
        "prune.py",
        "lowrank.py",
        "tokenizer.py",
        "size_report.py",
        "config.yaml",
        "README.md",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "artifact.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Include model payload plus code to emulate submission packaging.
            model_payload = model_payload_bytes(
                model,
                quant_bits=quant_bits,
                quant_group_size=quant_group_size,
                quant_exclude_patterns=quant_exclude_patterns,
                quant_fallback_dtype=quant_fallback_dtype,
            )
            zf.writestr("model_payload.bin", model_payload)
            zf.writestr("model_size_bytes.json", json.dumps({"model_bytes": len(model_payload)}))
            for rel in include:
                p = code_dir / rel
                if p.exists():
                    zf.write(p, arcname=rel)
        return zip_path.stat().st_size


def bytes_to_mb(n: int) -> float:
    return n / (1024.0 * 1024.0)
