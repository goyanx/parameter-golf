from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from lowrank import maybe_replace_linear
from model import ModelConfig, TinyTransformerLM
from prune import (
    global_sparsity,
    magnitude_prune_model,
    structured_prune_model,
)
from quantize import fake_quantize_model_inplace
from size_report import bytes_to_mb, estimate_artifact_bytes, parameter_count
from tokenizer import load_text, train_val_tokens

ARTIFACT_LIMIT_BYTES = 16 * 1024 * 1024


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_batch(tokens: List[int], batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= seq_len + 1:
        raise ValueError("Dataset too small for chosen seq_len")
    starts = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(tokens[s : s + seq_len], dtype=torch.long) for s in starts])
    y = torch.stack([torch.tensor(tokens[s + 1 : s + 1 + seq_len], dtype=torch.long) for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model: TinyTransformerLM, tokens: List[int], batch_size: int, seq_len: int, eval_steps: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(eval_steps):
        xb, yb = get_batch(tokens, batch_size, seq_len, device)
        _, loss = model(xb, yb)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def build_model(cfg: Dict) -> TinyTransformerLM:
    model_cfg = ModelConfig(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        attn_kv_heads=cfg["model"].get("attn_kv_heads"),
        n_layers=cfg["model"]["n_layers"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"].get("dropout", 0.0),
        weight_sharing=cfg["model"].get("weight_sharing", False),
        positional_encoding=cfg["model"].get("positional_encoding", "learned"),
    )
    model = TinyTransformerLM(model_cfg)
    lowrank_cfg = cfg.get("lowrank", {})
    rank = lowrank_cfg.get("rank")
    if rank is not None:
        model = maybe_replace_linear(
            model,
            rank=rank,
            include_patterns=_to_patterns(lowrank_cfg.get("include_patterns", [])),
            exclude_patterns=_to_patterns(lowrank_cfg.get("exclude_patterns", [])),
        )
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--run-name", default="baseline")
    return p.parse_args()

def _to_patterns(v) -> tuple[str, ...]:
    if not v:
        return ()
    if isinstance(v, str):
        return (v,)
    return tuple(str(x) for x in v)


def _to_layer_bits(v) -> tuple[tuple[str, int], ...]:
    if not v:
        return ()
    if isinstance(v, dict):
        items = v.items()
    elif isinstance(v, list):
        out = []
        for e in v:
            if not isinstance(e, dict):
                continue
            p = str(e.get("pattern", "")).strip()
            if not p:
                continue
            try:
                b = int(e.get("bits", 0))
            except (TypeError, ValueError):
                continue
            out.append((p, b))
        return tuple(out)
    else:
        return ()
    out = []
    for p, b in items:
        try:
            ib = int(b)
        except (TypeError, ValueError):
            continue
        out.append((str(p), ib))
    return tuple(out)

def _teacher_from_distill_cfg(cfg: Dict, model_cfg: Dict) -> Optional[Dict]:
    dcfg = cfg.get("distill", {})
    if not bool(dcfg.get("enabled", False)):
        return None
    teacher = dict(model_cfg)
    overrides = dcfg.get("teacher_model", {})
    teacher.update(overrides if isinstance(overrides, dict) else {})
    return teacher


def _distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
    t = max(temperature, 1e-6)
    s_logp = F.log_softmax(student_logits / t, dim=-1)
    t_p = F.softmax(teacher_logits / t, dim=-1)
    kd = F.kl_div(s_logp, t_p, reduction="batchmean") * (t * t)
    loss = alpha * ce + (1.0 - alpha) * kd
    return loss, ce.detach(), kd.detach()


def _interp_linear(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    frac = max(0.0, min(1.0, (step - 1) / float(total_steps - 1)))
    return start + frac * (end - start)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    root = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu")

    set_seed(int(cfg["train"].get("seed", 1337)))

    data_path = root / cfg["data"]["text_file"]
    if not data_path.exists():
        data_path.write_text(
            "Parameter golf toy dataset. Repeat useful text for small language model debugging.\n" * 512,
            encoding="utf-8",
        )

    text = load_text(data_path)
    splits = train_val_tokens(text, val_frac=float(cfg["data"].get("val_frac", 0.1)))
    cfg["model"]["vocab_size"] = 256

    model = build_model(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))

    batch_size = int(cfg["train"]["batch_size"])
    seq_len = int(cfg["model"]["max_seq_len"])
    steps = int(cfg["train"]["steps"])
    eval_every = int(cfg["train"].get("eval_every", 50))
    eval_steps = int(cfg["train"].get("eval_steps", 10))
    quant_bits = cfg.get("quantize", {}).get("bits")
    quant_group_size = int(cfg.get("quantize", {}).get("group_size", 64))
    quant_exclude_patterns = _to_patterns(cfg.get("quantize", {}).get("exclude_patterns", []))
    quant_fallback_dtype = str(cfg.get("quantize", {}).get("fallback_dtype", "fp16"))
    quant_pack_order = str(cfg.get("quantize", {}).get("pack_order", "state_dict"))
    quant_layer_bits = _to_layer_bits(cfg.get("quantize", {}).get("layer_bits", {}))
    distill_cfg = cfg.get("distill", {})
    distill_enabled = bool(distill_cfg.get("enabled", False))
    distill_alpha = float(distill_cfg.get("alpha", 0.5))
    distill_alpha_start = float(distill_cfg.get("alpha_start", distill_alpha))
    distill_alpha_end = float(distill_cfg.get("alpha_end", distill_alpha))
    distill_schedule = str(distill_cfg.get("alpha_schedule", "constant"))
    distill_temp = float(distill_cfg.get("temperature", 2.0))
    distill_hidden_enabled = bool(distill_cfg.get("hidden_enabled", False))
    distill_hidden_weight = float(distill_cfg.get("hidden_weight", 0.1))
    teacher_steps = int(distill_cfg.get("teacher_steps", 0))
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))

    prune_cfg = cfg.get("prune", {})
    prune_amount = float(prune_cfg.get("amount", 0.0))
    prune_mode = str(prune_cfg.get("mode", "magnitude")).lower()
    prune_include_patterns = _to_patterns(prune_cfg.get("include_patterns", []))
    prune_exclude_patterns = _to_patterns(prune_cfg.get("exclude_patterns", []))
    progressive_prune_enabled = bool(prune_cfg.get("progressive", False))
    progressive_prune_start = int(prune_cfg.get("start_step", max(1, steps // 2)))
    progressive_prune_end = int(prune_cfg.get("end_step", steps))
    progressive_prune_every = int(prune_cfg.get("every", 25))
    applied_prune_target = 0.0

    teacher_model: Optional[TinyTransformerLM] = None
    hidden_projector: Optional[torch.nn.Module] = None
    if distill_enabled:
        teacher_model_cfg = _teacher_from_distill_cfg(cfg, cfg["model"]) or cfg["model"]
        teacher_bundle = {"model": teacher_model_cfg, "lowrank": {"rank": None}}
        teacher_model = build_model(teacher_bundle).to(device)
        if distill_hidden_enabled:
            s_dim = int(cfg["model"]["d_model"])
            t_dim = int(teacher_model_cfg.get("d_model", s_dim))
            if t_dim != s_dim:
                hidden_projector = torch.nn.Linear(t_dim, s_dim, bias=False).to(device)
        teacher_opt = optim.AdamW(teacher_model.parameters(), lr=float(distill_cfg.get("teacher_lr", cfg["train"]["lr"])))
        for tstep in range(1, teacher_steps + 1):
            xb, yb = get_batch(splits["train"], batch_size, seq_len, device)
            _, tloss = teacher_model(xb, yb)
            teacher_opt.zero_grad(set_to_none=True)
            tloss.backward()
            teacher_opt.step()
            if eval_every > 0 and tstep % eval_every == 0:
                tval = eval_loss(teacher_model, splits["val"], batch_size, seq_len, eval_steps, device)
                print(f"[{args.run_name}] teacher_step={tstep} teacher_loss={tloss.item():.4f} teacher_val={tval:.4f}")
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        if hidden_projector is not None:
            for p in hidden_projector.parameters():
                p.requires_grad_(True)
            opt.add_param_group({"params": hidden_projector.parameters()})

    start = time.perf_counter()
    for step in range(1, steps + 1):
        xb, yb = get_batch(splits["train"], batch_size, seq_len, device)
        if distill_enabled and distill_hidden_enabled:
            slogits, ce_loss, shid = model(xb, yb, return_hidden=True)
        else:
            slogits, ce_loss = model(xb, yb)
            shid = None
        if distill_enabled and teacher_model is not None:
            with torch.no_grad():
                if distill_hidden_enabled:
                    tlogits, _, thid = teacher_model(xb, None, return_hidden=True)
                else:
                    tlogits, _ = teacher_model(xb, None)
                    thid = None
            if distill_schedule == "linear":
                alpha_t = _interp_linear(distill_alpha_start, distill_alpha_end, step, steps)
            else:
                alpha_t = distill_alpha
            loss, _, _ = _distill_loss(
                student_logits=slogits,
                teacher_logits=tlogits,
                targets=yb,
                alpha=alpha_t,
                temperature=distill_temp,
            )
            if distill_hidden_enabled:
                assert shid is not None and thid is not None
                t_h = thid
                s_h = shid
                if hidden_projector is not None:
                    t_h = hidden_projector(t_h)
                hidden_loss = F.mse_loss(s_h, t_h)
                loss = loss + distill_hidden_weight * hidden_loss
        else:
            loss = ce_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()

        if progressive_prune_enabled and prune_amount > 0 and step >= progressive_prune_start:
            if step % max(1, progressive_prune_every) == 0:
                span = max(1, progressive_prune_end - progressive_prune_start + 1)
                rel_step = min(span, step - progressive_prune_start + 1)
                target = prune_amount * (rel_step / span)
                if target > applied_prune_target:
                    # Apply only the incremental amount to avoid repeated over-pruning.
                    delta = max(0.0, min(0.95, target - applied_prune_target))
                    if delta > 0:
                        if prune_mode == "magnitude":
                            magnitude_prune_model(model, amount=delta)
                        else:
                            structured_prune_model(
                                model,
                                amount=delta,
                                mode=prune_mode,
                                include_patterns=prune_include_patterns,
                                exclude_patterns=prune_exclude_patterns,
                            )
                        applied_prune_target = target

        if eval_every > 0 and step % eval_every == 0:
            val = eval_loss(model, splits["val"], batch_size, seq_len, eval_steps, device)
            print(f"[{args.run_name}] step={step} train_loss={loss.item():.4f} val_loss={val:.4f}")

    prune_stats = {}
    if prune_amount > 0 and not progressive_prune_enabled:
        if prune_mode == "magnitude":
            prune_stats = magnitude_prune_model(model, amount=prune_amount)
        else:
            prune_stats = structured_prune_model(
                model,
                amount=prune_amount,
                mode=prune_mode,
                include_patterns=prune_include_patterns,
                exclude_patterns=prune_exclude_patterns,
            )
    elif prune_amount > 0 and progressive_prune_enabled:
        # Collect layer stats with a no-op prune call.
        if prune_mode == "magnitude":
            prune_stats = {k: v for k, v in magnitude_prune_model(model, amount=0.0).items()}
        else:
            prune_stats = {
                k: v
                for k, v in structured_prune_model(
                    model,
                    amount=0.0,
                    mode=prune_mode,
                    include_patterns=prune_include_patterns,
                    exclude_patterns=prune_exclude_patterns,
                ).items()
            }

    qat_cfg = cfg.get("qat", {})
    qat_enabled = bool(qat_cfg.get("enabled", False))
    qat_steps = int(qat_cfg.get("steps", 0))
    qat_lr = float(qat_cfg.get("lr", cfg["train"]["lr"]))
    qat_grad_clip = float(qat_cfg.get("grad_clip", grad_clip))
    qat_delay_fake_quant_steps = int(qat_cfg.get("delay_fake_quant_steps", 0))
    if quant_bits is not None and qat_enabled and qat_steps > 0:
        qat_opt = optim.AdamW(model.parameters(), lr=qat_lr)
        for qstep in range(1, qat_steps + 1):
            if qstep > qat_delay_fake_quant_steps:
                fake_quantize_model_inplace(
                    model,
                    bits=int(quant_bits),
                    group_size=quant_group_size,
                    exclude_patterns=quant_exclude_patterns or ("token_emb", "head", "ln", "norm"),
                    layer_bits=quant_layer_bits,
                )
            xb, yb = get_batch(splits["train"], batch_size, seq_len, device)
            _, loss = model(xb, yb)
            qat_opt.zero_grad(set_to_none=True)
            loss.backward()
            if qat_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=qat_grad_clip)
            qat_opt.step()

    train_time = time.perf_counter() - start
    val_loss = eval_loss(model, splits["val"], batch_size, seq_len, eval_steps, device)

    artifact_bytes = estimate_artifact_bytes(
        model,
        root,
        quant_bits=quant_bits,
        quant_group_size=quant_group_size,
        quant_exclude_patterns=quant_exclude_patterns or ("token_emb", "head", "ln", "norm"),
        quant_fallback_dtype=quant_fallback_dtype,
        quant_pack_order=quant_pack_order,
        quant_layer_bits=quant_layer_bits,
    )
    params = parameter_count(model)
    nonzero, total, sparsity = global_sparsity(model)

    if bool(int(os.environ.get("SAVE_MODEL_STATE", "0"))):
        save_path = Path(os.environ.get("SAVE_MODEL_STATE_PATH", str(root / "final_model.pt")))
        torch.save(model.state_dict(), save_path)

    if artifact_bytes > ARTIFACT_LIMIT_BYTES:
        raise RuntimeError(
            f"Artifact too large: {bytes_to_mb(artifact_bytes):.4f} MB > {bytes_to_mb(ARTIFACT_LIMIT_BYTES):.4f} MB limit"
        )

    metrics = {
        "run_name": args.run_name,
        "parameter_count": params,
        "nonzero_parameters": nonzero,
        "total_parameters": total,
        "global_sparsity": sparsity,
        "artifact_size_bytes": artifact_bytes,
        "artifact_size_mb": bytes_to_mb(artifact_bytes),
        "train_time_sec": train_time,
        "val_loss": val_loss,
        "prune_layers": len(prune_stats),
        "prune_mode": prune_mode,
        "prune_include_patterns": list(prune_include_patterns),
        "prune_exclude_patterns": list(prune_exclude_patterns),
        "quant_bits": quant_bits,
        "quant_group_size": quant_group_size,
        "attn_kv_heads": cfg["model"].get("attn_kv_heads"),
        "positional_encoding": cfg["model"].get("positional_encoding", "learned"),
        "quant_exclude_patterns": list(quant_exclude_patterns),
        "quant_pack_order": quant_pack_order,
        "quant_layer_bits": [{"pattern": p, "bits": b} for p, b in quant_layer_bits],
        "distill_enabled": distill_enabled,
        "distill_alpha": distill_alpha,
        "distill_alpha_start": distill_alpha_start,
        "distill_alpha_end": distill_alpha_end,
        "distill_schedule": distill_schedule,
        "distill_temperature": distill_temp,
        "distill_hidden_enabled": distill_hidden_enabled,
        "distill_hidden_weight": distill_hidden_weight,
        "teacher_steps": teacher_steps,
        "grad_clip": grad_clip,
        "progressive_prune_enabled": progressive_prune_enabled,
        "progressive_prune_target": prune_amount,
        "qat_enabled": qat_enabled,
        "qat_steps": qat_steps,
        "qat_delay_fake_quant_steps": qat_delay_fake_quant_steps,
        "qat_grad_clip": qat_grad_clip,
        "device": str(device),
    }

    out_dir = root / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_name}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
