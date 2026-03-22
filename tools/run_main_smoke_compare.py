from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


FINAL_EXACT_RE = re.compile(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)")
SIZE_RE = re.compile(r"Total submission size int8\+zlib: (\d+) bytes")
STOP_RE = re.compile(r"stopping_early: wallclock_cap .*step:(\d+)/")


def run_case(name: str, env_overrides: dict[str, str], log_dir: Path, python_exe: str) -> dict:
    log_path = log_dir / f"{name}.log"
    env = os.environ.copy()
    env.update(env_overrides)

    cmd = [python_exe, "train_gpt.py"]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(out, encoding="utf-8")

    result = {"name": name, "status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode, "log": str(log_path)}
    m = FINAL_EXACT_RE.search(out)
    if m:
        result["val_loss"] = float(m.group(1))
        result["val_bpb"] = float(m.group(2))
    m = SIZE_RE.search(out)
    if m:
        result["submission_size_bytes"] = int(m.group(1))
    m = STOP_RE.search(out)
    if m:
        result["stop_step"] = int(m.group(1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run train_gpt smoke compare: baseline vs toy_micro_best preset.")
    ap.add_argument("--max-wallclock-seconds", type=float, default=120.0)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--run-id-prefix", default="smoke")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs / f"main_smoke_compare_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "MAX_WALLCLOCK_SECONDS": str(args.max_wallclock_seconds),
        "ITERATIONS": str(args.iterations),
    }

    baseline_env = {
        **common,
        "RUN_ID": f"{args.run_id_prefix}_baseline_{ts}",
        "COMPRESSION_PRESET": "",
    }
    preset_env = {
        **common,
        "RUN_ID": f"{args.run_id_prefix}_toy_micro_best_{ts}",
        "COMPRESSION_PRESET": "toy_micro_best",
    }

    python_exe = sys.executable
    baseline = run_case("baseline", baseline_env, run_dir, python_exe)
    preset = run_case("toy_micro_best", preset_env, run_dir, python_exe)

    summary = {"timestamp": ts, "baseline": baseline, "toy_micro_best": preset}
    if baseline.get("status") == "ok" and preset.get("status") == "ok":
        if "val_bpb" in baseline and "val_bpb" in preset:
            summary["delta_val_bpb"] = preset["val_bpb"] - baseline["val_bpb"]
        if "val_loss" in baseline and "val_loss" in preset:
            summary["delta_val_loss"] = preset["val_loss"] - baseline["val_loss"]
        if "submission_size_bytes" in baseline and "submission_size_bytes" in preset:
            summary["delta_submission_size_bytes"] = preset["submission_size_bytes"] - baseline["submission_size_bytes"]

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

