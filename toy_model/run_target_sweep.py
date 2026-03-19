from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


def write_config(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def run_case(root: Path, name: str, cfg: dict, python_exe: str) -> dict:
    cfg_path = root / f"tmp_{name}.yaml"
    write_config(cfg_path, cfg)
    cmd = [python_exe, str(root / "train.py"), "--config", str(cfg_path), "--run-name", name]
    subprocess.run(cmd, check=True)
    out = root / "runs" / f"{name}.json"
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parent
    python_exe = sys.executable
    base = yaml.safe_load((root / "config_distill_qat.yaml").read_text(encoding="utf-8"))

    cases = []

    c = copy.deepcopy(base)
    c["prune"]["amount"] = 0.15
    c["lowrank"]["rank"] = 48
    c["qat"]["steps"] = 50
    c["distill"]["alpha"] = 0.4
    c["distill"]["teacher_steps"] = 120
    cases.append(("target_a", c))

    c = copy.deepcopy(base)
    c["prune"]["amount"] = 0.12
    c["lowrank"]["rank"] = 64
    c["qat"]["steps"] = 60
    c["distill"]["alpha"] = 0.45
    c["distill"]["teacher_steps"] = 140
    cases.append(("target_b", c))

    c = copy.deepcopy(base)
    c["prune"]["amount"] = 0.2
    c["lowrank"]["rank"] = 48
    c["qat"]["steps"] = 70
    c["distill"]["alpha"] = 0.35
    c["distill"]["teacher_steps"] = 180
    cases.append(("target_c", c))

    c = copy.deepcopy(base)
    c["prune"]["amount"] = 0.08
    c["lowrank"]["rank"] = 64
    c["qat"]["steps"] = 80
    c["distill"]["alpha"] = 0.5
    c["distill"]["teacher_steps"] = 150
    cases.append(("target_d", c))

    results = []
    for name, cfg in cases:
        print(f"Running {name} ...")
        metrics = run_case(root, name, cfg, python_exe=python_exe)
        results.append(metrics)

    def score(m: dict) -> tuple:
        return (m["val_loss"], abs(m["artifact_size_mb"] - 0.12))

    target_hits = [m for m in results if m["artifact_size_mb"] <= 0.15 and m["val_loss"] < 2.8]
    target_hits.sort(key=score)
    results.sort(key=score)

    summary = {
        "best_overall": results[0] if results else None,
        "target_hits": target_hits,
        "all_results": results,
    }
    out = root / "runs" / "target_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()

