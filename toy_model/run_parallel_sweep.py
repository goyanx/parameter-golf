from __future__ import annotations

import copy
import itertools
import json
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def run_one(root: Path, python_exe: str, name: str, cfg: dict) -> dict:
    tmp_cfg = root / f"tmp_{name}.yaml"
    log_dir = root / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    write_yaml(tmp_cfg, cfg)
    cmd = [python_exe, str(root / "train.py"), "--config", str(tmp_cfg), "--run-name", name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")

    if proc.returncode != 0:
        return {"run_name": name, "status": "failed", "returncode": proc.returncode, "log": str(log_path)}

    metrics_path = root / "runs" / f"{name}.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["status"] = "ok"
    metrics["log"] = str(log_path)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--mode", choices=["light", "mixed", "chassis", "frontier", "full"], default="light")
    return p.parse_args()


def build_cases(base_cfg: dict, mode: str) -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []
    idx = 0

    # Keep lowrank disabled for quality recovery.
    # Light mode is intentionally small for desktop stability.
    if mode == "light":
        prune_vals = [0.05, 0.10]
        qat_steps_vals = [60, 90]
        alpha_vals = [0.35, 0.45]
        teacher_steps_vals = [180]
        lowrank_vals = [None]
    elif mode == "mixed":
        # Mild structural compression + strong recovery.
        prune_vals = [0.12, 0.18]
        qat_steps_vals = [90]
        alpha_vals = [0.35, 0.45]
        teacher_steps_vals = [220]
        lowrank_vals = [48, 64]
        group_size_vals = [64]
        delay_fake_quant_vals = [0]
    elif mode == "chassis":
        # Chassis tuning: quantizer group granularity + delayed fake quant.
        prune_vals = [0.12]
        qat_steps_vals = [90]
        alpha_vals = [0.35, 0.45]
        teacher_steps_vals = [220]
        lowrank_vals = [48]
        group_size_vals = [32, 48, 64]
        delay_fake_quant_vals = [0, 20]
        temp_vals = [2.5]
    elif mode == "frontier":
        # Last-mile search near current Pareto frontier.
        prune_vals = [0.10, 0.12]
        qat_steps_vals = [90]
        alpha_vals = [0.35]
        teacher_steps_vals = [220]
        lowrank_vals = [56, 64]
        group_size_vals = [32, 48]
        delay_fake_quant_vals = [20]
        temp_vals = [2.0, 2.5]
    else:
        prune_vals = [0.05, 0.10, 0.15]
        qat_steps_vals = [60, 90]
        alpha_vals = [0.30, 0.40, 0.50]
        teacher_steps_vals = [180, 240]
        lowrank_vals = [None]
        group_size_vals = [64]
        delay_fake_quant_vals = [0]
        temp_vals = [2.5]

    grid = itertools.product(
        prune_vals,
        qat_steps_vals,
        alpha_vals,
        teacher_steps_vals,
        lowrank_vals,
        group_size_vals,
        delay_fake_quant_vals,
        temp_vals,
    )
    for prune_amt, qat_steps, alpha, teacher_steps, lowrank_rank, group_size, delay_fake_quant, temp in grid:
        idx += 1
        name = f"par_{idx:02d}"
        cfg = copy.deepcopy(base_cfg)

        cfg["lowrank"]["rank"] = lowrank_rank
        cfg["prune"]["amount"] = prune_amt
        cfg["prune"]["progressive"] = True
        cfg["prune"]["start_step"] = 100
        cfg["prune"]["end_step"] = 200
        cfg["prune"]["every"] = 25
        cfg["qat"]["enabled"] = True
        cfg["qat"]["steps"] = qat_steps
        cfg["qat"]["lr"] = 2e-4
        cfg["qat"]["grad_clip"] = 0.8
        cfg["qat"]["delay_fake_quant_steps"] = delay_fake_quant
        cfg["quantize"]["group_size"] = group_size

        cfg["distill"]["enabled"] = True
        cfg["distill"]["alpha"] = alpha
        cfg["distill"]["alpha_start"] = min(0.7, alpha + 0.15)
        cfg["distill"]["alpha_end"] = max(0.2, alpha - 0.1)
        cfg["distill"]["alpha_schedule"] = "linear"
        cfg["distill"]["temperature"] = temp
        cfg["distill"]["teacher_steps"] = teacher_steps
        cfg["distill"]["teacher_lr"] = 1e-3
        cfg["distill"]["teacher_model"]["d_model"] = 256
        cfg["distill"]["teacher_model"]["n_heads"] = 8
        cfg["distill"]["teacher_model"]["n_layers"] = 4
        cfg["distill"]["teacher_model"]["weight_sharing"] = False

        # Keep training-time bounded for local iteration.
        cfg["train"]["steps"] = 200
        cfg["train"]["eval_every"] = 50
        cfg["train"]["eval_steps"] = 8
        cfg["train"]["grad_clip"] = 0.8

        cases.append((name, cfg))

    return cases


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    python_exe = sys.executable
    base_cfg = yaml.safe_load((root / "config_distill_qat.yaml").read_text(encoding="utf-8"))
    cases = build_cases(base_cfg, mode=args.mode)

    max_workers = max(1, int(args.max_workers))
    results: list[dict] = []
    print(f"Running {len(cases)} cases with max_workers={max_workers}, mode={args.mode}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, root, python_exe, name, cfg) for name, cfg in cases]
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            rn = res.get("run_name", "unknown")
            st = res.get("status", "unknown")
            if st == "ok":
                print(
                    f"{rn}: ok size_mb={res['artifact_size_mb']:.4f} "
                    f"val_loss={res['val_loss']:.4f} time={res['train_time_sec']:.2f}s"
                )
            else:
                print(f"{rn}: failed (see {res.get('log')})")

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]

    ok_sorted = sorted(ok, key=lambda r: (r["val_loss"], abs(r["artifact_size_mb"] - 0.12)))
    size_focus = sorted(ok, key=lambda r: (abs(r["artifact_size_mb"] - 0.12), r["val_loss"]))
    target_hits = [r for r in ok if r["artifact_size_mb"] <= 0.15 and r["val_loss"] < 2.8]
    target_hits = sorted(target_hits, key=lambda r: (r["val_loss"], r["artifact_size_mb"]))

    summary = {
        "total_cases": len(cases),
        "completed": len(ok),
        "failed": len(failed),
        "best_loss": ok_sorted[0] if ok_sorted else None,
        "closest_to_012mb": size_focus[0] if size_focus else None,
        "target_hits": target_hits,
        "top10_by_loss": ok_sorted[:10],
        "all_results": sorted(ok, key=lambda r: r["run_name"]) + failed,
    }
    out = root / "runs" / "parallel_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
