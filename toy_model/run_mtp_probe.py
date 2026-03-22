from __future__ import annotations

import copy
import json
import os
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
    out = root / "runs" / f"{name}.json"
    m = json.loads(out.read_text(encoding="utf-8"))
    m["status"] = "ok"
    m["log"] = str(log_path)
    return m


def main() -> None:
    root = Path(__file__).resolve().parent
    python_exe = os.environ.get("PYTHON_EXE", sys.executable)
    max_workers = max(1, int(os.environ.get("MAX_WORKERS", "2")))
    base = yaml.safe_load((root / "config_toyfocus_best.yaml").read_text(encoding="utf-8"))

    variants = []
    c0 = copy.deepcopy(base)
    c0["mtp"]["enabled"] = False
    variants.append(("mtp_base", c0))

    c1 = copy.deepcopy(base)
    c1["mtp"]["enabled"] = True
    c1["mtp"]["weight"] = 0.10
    c1["mtp"]["horizons"] = [2]
    variants.append(("mtp_h2_w010", c1))

    c2 = copy.deepcopy(base)
    c2["mtp"]["enabled"] = True
    c2["mtp"]["weight"] = 0.20
    c2["mtp"]["horizons"] = [2]
    variants.append(("mtp_h2_w020", c2))

    c3 = copy.deepcopy(base)
    c3["mtp"]["enabled"] = True
    c3["mtp"]["weight"] = 0.10
    c3["mtp"]["horizons"] = [2, 3]
    variants.append(("mtp_h23_w010", c3))

    print(f"Running {len(variants)} MTP cases with max_workers={max_workers}")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, root, python_exe, name, cfg) for name, cfg in variants]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r.get("status") == "ok":
                print(
                    f"{r['run_name']}: size_mb={r['artifact_size_mb']:.4f} val_loss={r['val_loss']:.4f} "
                    f"mtp={r.get('mtp_enabled')} horizons={r.get('mtp_horizons')}"
                )
            else:
                print(f"{r.get('run_name')}: failed")

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]
    by_loss = sorted(ok, key=lambda r: (r["val_loss"], r["artifact_size_mb"]))
    base_m = next((r for r in ok if r["run_name"] == "mtp_base"), None)
    deltas = []
    if base_m is not None:
        for r in ok:
            if r["run_name"] == "mtp_base":
                continue
            deltas.append(
                {
                    "run_name": r["run_name"],
                    "delta_val_loss": r["val_loss"] - base_m["val_loss"],
                    "delta_artifact_size_mb": r["artifact_size_mb"] - base_m["artifact_size_mb"],
                    "mtp_weight": r.get("mtp_weight"),
                    "mtp_horizons": r.get("mtp_horizons"),
                }
            )
        deltas.sort(key=lambda d: (d["delta_val_loss"], d["delta_artifact_size_mb"]))

    summary = {
        "total_cases": len(variants),
        "completed": len(ok),
        "failed": len(failed),
        "baseline": base_m,
        "best_loss": by_loss[0] if by_loss else None,
        "deltas_vs_base": deltas,
        "all_results": sorted(ok, key=lambda r: r["run_name"]) + failed,
    }
    out = root / "runs" / "mtp_probe_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
