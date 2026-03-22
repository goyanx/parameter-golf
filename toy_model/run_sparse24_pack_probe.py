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
    variants.append(("sparse24_base", v0))

    v1 = copy.deepcopy(base)
    v1["prune"]["mode"] = "nm2_4"
    v1["prune"]["progressive"] = False
    v1["prune"]["include_patterns"] = ["attn", "mlp"]
    v1["qat"]["enabled"] = False
    v1["quantize"]["sparse_2_4_pack"] = False
    variants.append(("sparse24_nm_nopack", v1))

    v2 = copy.deepcopy(v1)
    v2["quantize"]["sparse_2_4_pack"] = True
    variants.append(("sparse24_nm_pack", v2))

    v3 = copy.deepcopy(base)
    v3["prune"]["mode"] = "nm2_4"
    v3["prune"]["progressive"] = False
    v3["prune"]["include_patterns"] = ["attn"]
    v3["quantize"]["sparse_2_4_pack"] = False
    variants.append(("sparse24_nm_attn_nopack", v3))

    v4 = copy.deepcopy(v3)
    v4["quantize"]["sparse_2_4_pack"] = True
    variants.append(("sparse24_nm_attn_pack", v4))

    results = []
    for name, cfg in variants:
        m = run_one(root, python_exe, name, cfg)
        results.append(m)
        print(f"{name}: size_mb={m['artifact_size_mb']:.4f} val_loss={m['val_loss']:.4f}")

    base_m = next(r for r in results if r["run_name"] == "sparse24_base")
    deltas = []
    for r in results:
        if r["run_name"] == "sparse24_base":
            continue
        deltas.append(
            {
                "run_name": r["run_name"],
                "delta_val_loss": r["val_loss"] - base_m["val_loss"],
                "delta_artifact_size_mb": r["artifact_size_mb"] - base_m["artifact_size_mb"],
                "prune_mode": r.get("prune_mode"),
                "quant_sparse_2_4_pack": r.get("quant_sparse_2_4_pack"),
            }
        )

    summary = {"results": results, "deltas_vs_base": deltas}
    out = root / "runs" / "sparse24_pack_probe_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
