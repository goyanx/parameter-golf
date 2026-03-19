from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def write_config(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def run_case(root: Path, name: str, cfg: dict) -> dict:
    cfg_path = root / "tmp_config.yaml"
    write_config(cfg_path, cfg)
    cmd = [sys.executable, str(root / "train.py"), "--config", str(cfg_path), "--run-name", name]
    subprocess.run(cmd, check=True)
    metrics_path = root / "runs" / f"{name}.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parent
    base_cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))

    cases = []

    c = json.loads(json.dumps(base_cfg))
    c["prune"]["amount"] = 0.0
    c["quantize"]["bits"] = None
    c["quantize"]["group_size"] = 64
    c["model"]["weight_sharing"] = False
    c["lowrank"]["rank"] = None
    c["qat"]["enabled"] = False
    c["qat"]["steps"] = 0
    cases.append(("baseline", c))

    c = json.loads(json.dumps(base_cfg))
    c["quantize"]["bits"] = 4
    c["qat"]["enabled"] = True
    c["qat"]["steps"] = 30
    c["qat"]["lr"] = 3e-4
    cases.append(("quant4_qat", c))

    c = json.loads(json.dumps(base_cfg))
    c["prune"]["amount"] = 0.3
    cases.append(("prune30", c))

    c = json.loads(json.dumps(base_cfg))
    c["model"]["weight_sharing"] = True
    cases.append(("weight_share", c))

    c = json.loads(json.dumps(base_cfg))
    c["lowrank"]["rank"] = 32
    cases.append(("lowrank32", c))

    results = []
    for name, cfg in cases:
        print(f"Running ablation: {name}")
        metrics = run_case(root, name, cfg)
        results.append(metrics)

    summary_path = root / "runs" / "ablation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved ablation summary: {summary_path}")


if __name__ == "__main__":
    main()
