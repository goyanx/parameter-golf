from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

from quantize import fake_quantize_model_inplace
from size_report import estimate_artifact_bytes
from train import build_model, eval_loss, load_text, set_seed, train_val_tokens


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    n = name.lower()
    return any(p.lower() in n for p in patterns)


def _module_prefix(param_name: str) -> str:
    parts = param_name.split(".")
    if len(parts) <= 1:
        return param_name
    return ".".join(parts[:-1])


def run_one(root: Path, python_exe: str, name: str, cfg: dict) -> dict:
    tmp_cfg = root / f"tmp_{name}.yaml"
    log_dir = root / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    write_yaml(tmp_cfg, cfg)
    env = os.environ.copy()
    env["SAVE_MODEL_STATE"] = "1"
    env["SAVE_MODEL_STATE_PATH"] = str(root / f"{name}_state.pt")
    cmd = [python_exe, str(root / "train.py"), "--config", str(tmp_cfg), "--run-name", name]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")

    if proc.returncode != 0:
        return {"run_name": name, "status": "failed", "returncode": proc.returncode, "log": str(log_path)}

    metrics_path = root / "runs" / f"{name}.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["status"] = "ok"
    metrics["log"] = str(log_path)
    return metrics


def train_baseline(root: Path, python_exe: str, cfg: dict) -> dict:
    baseline_cfg = copy.deepcopy(cfg)
    return run_one(root, python_exe, "precision_ranked_base", baseline_cfg)


def load_trained_model(root: Path, cfg: dict, state_path: Path) -> torch.nn.Module:
    model = build_model(cfg)
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def candidate_prefixes(model: torch.nn.Module, base_exclude: tuple[str, ...]) -> list[str]:
    prefixes: set[str] = set()
    for name, p in model.named_parameters():
        if not torch.is_floating_point(p):
            continue
        prefix = _module_prefix(name)
        if _matches_any(prefix, base_exclude):
            continue
        prefixes.add(prefix)
    return sorted(prefixes)


def evaluate_keep_set(
    model: torch.nn.Module,
    root: Path,
    cfg: dict,
    keep_prefixes: tuple[str, ...],
    base_exclude: tuple[str, ...],
    val_tokens,
    batch_size: int,
    seq_len: int,
    eval_steps: int,
    device: torch.device,
) -> dict:
    quant_cfg = cfg.get("quantize", {})
    bits = int(quant_cfg.get("bits", 4))
    group_size = int(quant_cfg.get("group_size", 64))
    fallback_dtype = str(quant_cfg.get("fallback_dtype", "fp16"))
    pack_order = str(quant_cfg.get("pack_order", "state_dict"))
    exclude = tuple(base_exclude) + keep_prefixes

    qmodel = copy.deepcopy(model)
    fake_quantize_model_inplace(
        qmodel,
        bits=bits,
        group_size=group_size,
        exclude_patterns=exclude,
    )
    loss = eval_loss(qmodel, val_tokens, batch_size, seq_len, eval_steps, device)
    artifact_bytes = estimate_artifact_bytes(
        qmodel,
        root,
        quant_bits=bits,
        quant_group_size=group_size,
        quant_exclude_patterns=exclude,
        quant_fallback_dtype=fallback_dtype,
        quant_pack_order=pack_order,
    )
    return {
        "keep_prefixes": list(keep_prefixes),
        "val_loss": loss,
        "artifact_size_bytes": artifact_bytes,
        "artifact_size_mb": artifact_bytes / (1024.0 * 1024.0),
    }


def build_precision_cfg(base_cfg: dict, keep_prefixes: list[str]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    ex = list(cfg.get("quantize", {}).get("exclude_patterns", []))
    for prefix in keep_prefixes:
        if prefix not in ex:
            ex.append(prefix)
    cfg["quantize"]["exclude_patterns"] = ex
    cfg["quantize"]["fallback_dtype"] = "fp16"
    return cfg


def main() -> None:
    root = Path(__file__).resolve().parent
    python_exe = os.environ.get("PYTHON_EXE", sys.executable)
    base_cfg = yaml.safe_load((root / "config_micro_best.yaml").read_text(encoding="utf-8"))

    baseline_metrics = train_baseline(root, python_exe, base_cfg)
    if baseline_metrics.get("status") != "ok":
        raise RuntimeError(f"Baseline training failed: {baseline_metrics.get('log')}")

    state_path = root / "precision_ranked_base_state.pt"
    model = load_trained_model(root, base_cfg, state_path)
    set_seed(int(base_cfg["train"].get("seed", 1337)))
    data_path = root / base_cfg["data"]["text_file"]
    text = load_text(data_path)
    splits = train_val_tokens(text, val_frac=float(base_cfg["data"].get("val_frac", 0.1)))
    batch_size = int(base_cfg["train"]["batch_size"])
    seq_len = int(base_cfg["model"]["max_seq_len"])
    eval_steps = int(base_cfg["train"]["eval_steps"])
    device = torch.device("cuda" if torch.cuda.is_available() and base_cfg["train"].get("use_cuda", True) else "cpu")
    val_tokens = splits["val"]

    quant_cfg = base_cfg["quantize"]
    base_exclude = tuple(quant_cfg.get("exclude_patterns", []))
    pack_order = str(quant_cfg.get("pack_order", "state_dict"))

    # Baseline full-quantized reference for ranking.
    baseline_qmodel = copy.deepcopy(model)
    fake_quantize_model_inplace(
        baseline_qmodel,
        bits=int(quant_cfg.get("bits", 4)),
        group_size=int(quant_cfg.get("group_size", 64)),
        exclude_patterns=base_exclude,
    )
    baseline_quant_loss = eval_loss(baseline_qmodel, val_tokens, batch_size, seq_len, eval_steps, device)
    baseline_quant_bytes = estimate_artifact_bytes(
        baseline_qmodel,
        root,
        quant_bits=int(quant_cfg.get("bits", 4)),
        quant_group_size=int(quant_cfg.get("group_size", 64)),
        quant_exclude_patterns=base_exclude,
        quant_fallback_dtype=str(quant_cfg.get("fallback_dtype", "fp16")),
        quant_pack_order=pack_order,
    )

    prefixes = candidate_prefixes(model, base_exclude)

    def score_prefix(prefix: str) -> dict:
        result = evaluate_keep_set(
            model=model,
            root=root,
            cfg=base_cfg,
            keep_prefixes=(prefix,),
            base_exclude=base_exclude,
            val_tokens=val_tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            eval_steps=eval_steps,
            device=device,
        )
        result["prefix"] = prefix
        result["loss_improvement"] = baseline_quant_loss - result["val_loss"]
        result["bytes_added"] = max(0, result["artifact_size_bytes"] - baseline_quant_bytes)
        result["improvement_per_byte"] = result["loss_improvement"] / max(result["bytes_added"], 1)
        return result

    sensitivity_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(os.environ.get("MAX_WORKERS", "2")))) as ex:
        futs = [ex.submit(score_prefix, prefix) for prefix in prefixes]
        for fut in as_completed(futs):
            sensitivity_results.append(fut.result())

    sensitivity_results.sort(key=lambda r: (r["improvement_per_byte"], r["loss_improvement"]), reverse=True)
    top_prefixes = [r["prefix"] for r in sensitivity_results[:3] if r["loss_improvement"] > 0]
    if not top_prefixes:
        top_prefixes = [r["prefix"] for r in sensitivity_results[:1]]

    ranked_cfgs: list[tuple[str, dict]] = []
    for n in range(1, min(3, len(top_prefixes)) + 1):
        keep = top_prefixes[:n]
        cfg = build_precision_cfg(base_cfg, keep)
        ranked_cfgs.append((f"precision_ranked_{n}", cfg))

    ranked_results: list[dict] = []
    if ranked_cfgs:
        with ThreadPoolExecutor(max_workers=max(1, int(os.environ.get("MAX_WORKERS", "2")))) as ex:
            futs = [ex.submit(run_one, root, python_exe, name, cfg) for name, cfg in ranked_cfgs]
            for fut in as_completed(futs):
                ranked_results.append(fut.result())

    summary = {
        "baseline_metrics": baseline_metrics,
        "baseline_quant_loss": baseline_quant_loss,
        "baseline_quant_bytes": baseline_quant_bytes,
        "sensitivity_results": sensitivity_results,
        "top_prefixes": top_prefixes,
        "ranked_results": ranked_results,
    }
    out = root / "runs" / "precision_ranked_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")
    print(f"Top prefixes: {top_prefixes}")


if __name__ == "__main__":
    main()
