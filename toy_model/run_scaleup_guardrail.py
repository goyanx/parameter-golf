from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a quick baseline vs scale-up guardrail check.")
    p.add_argument("--steps", type=int, default=60, help="Override train steps for quick checks")
    p.add_argument("--max-seconds", type=float, default=30.0, help="Per-run budget simulation")
    return p.parse_args()


def _run_case(root: Path, cfg_path: Path, run_name: str, steps: int, max_seconds: float) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("train", {})
    cfg["train"]["steps"] = steps
    cfg["train"]["time_budget_sec"] = max_seconds
    cfg["train"]["stop_on_budget"] = True
    cfg["train"]["budget_check_every"] = 5
    cfg["train"]["eval_every"] = max(10, min(steps, int(cfg["train"].get("eval_every", 50))))
    cfg["train"]["eval_steps"] = int(cfg["train"].get("eval_steps", 6))

    tmp_cfg = root / f"tmp_{run_name}.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cmd = [sys.executable, str(root / "train.py"), "--config", str(tmp_cfg), "--run-name", run_name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path = root / "runs" / "logs" / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")

    if proc.returncode != 0:
        return {
            "run_name": run_name,
            "status": "failed",
            "returncode": proc.returncode,
            "config": str(cfg_path),
            "log": str(log_path),
        }

    metrics = json.loads((root / "runs" / f"{run_name}.json").read_text(encoding="utf-8"))
    metrics["status"] = "ok"
    metrics["config"] = str(cfg_path)
    metrics["log"] = str(log_path)
    return metrics


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    cases = [
        ("guard_base", root / "config.yaml"),
        ("guard_scaleup", root / "config_scaleup_desktop.yaml"),
    ]

    results = [_run_case(root, cfg, name, steps=args.steps, max_seconds=args.max_seconds) for name, cfg in cases]
    ok = [r for r in results if r.get("status") == "ok"]

    summary = {
        "steps": args.steps,
        "max_seconds": args.max_seconds,
        "results": results,
        "artifact_delta_mb": (ok[1]["artifact_size_mb"] - ok[0]["artifact_size_mb"]) if len(ok) == 2 else None,
        "loss_delta": (ok[1]["val_loss"] - ok[0]["val_loss"]) if len(ok) == 2 else None,
    }

    out = root / "runs" / "scaleup_guardrail_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
