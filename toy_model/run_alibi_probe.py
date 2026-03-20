from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def run_one(root: Path, python_exe: str, name: str, cfg: dict) -> dict:
    tmp_cfg = root / f"tmp_{name}.yaml"
    write_yaml(tmp_cfg, cfg)
    cmd = [python_exe, str(root / "train.py"), "--config", str(tmp_cfg), "--run-name", name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed:\n{proc.stdout}\n{proc.stderr}")
    out = root / "runs" / f"{name}.json"
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parent
    python_exe = os.environ.get("PYTHON_EXE", sys.executable)
    base = yaml.safe_load((root / "config_toyfocus_best.yaml").read_text(encoding="utf-8"))

    variants: list[tuple[str, dict]] = []
    v0 = copy.deepcopy(base)
    v0["model"]["positional_encoding"] = "learned"
    variants.append(("alibi_probe_learned", v0))

    v1 = copy.deepcopy(base)
    v1["model"]["positional_encoding"] = "alibi"
    variants.append(("alibi_probe_alibi", v1))

    results = []
    for name, cfg in variants:
        m = run_one(root, python_exe, name, cfg)
        results.append(m)
        print(
            f"{name}: pos={m.get('positional_encoding', cfg['model']['positional_encoding'])} "
            f"size_mb={m['artifact_size_mb']:.4f} val_loss={m['val_loss']:.4f}"
        )

    learned = next(r for r in results if r["run_name"] == "alibi_probe_learned")
    alibi = next(r for r in results if r["run_name"] == "alibi_probe_alibi")
    summary = {
        "results": results,
        "delta_alibi_vs_learned": {
            "delta_val_loss": alibi["val_loss"] - learned["val_loss"],
            "delta_artifact_size_mb": alibi["artifact_size_mb"] - learned["artifact_size_mb"],
        },
    }

    out = root / "runs" / "alibi_probe_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
