from __future__ import annotations

import copy
import json
import os
import statistics
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

    seeds = [1337, 2027]
    cases = []
    for seed in seeds:
        cfg = copy.deepcopy(base)
        cfg["train"]["seed"] = int(seed)
        cases.append((f"seedcheck_{seed}", cfg))

    print(f"Running {len(cases)} seed checks with max_workers={max_workers}")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, root, python_exe, name, cfg) for name, cfg in cases]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r.get("status") == "ok":
                print(f"{r['run_name']}: size_mb={r['artifact_size_mb']:.4f} val_loss={r['val_loss']:.4f}")
            else:
                print(f"{r.get('run_name')}: failed")

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]

    losses = [r["val_loss"] for r in ok]
    sizes = [r["artifact_size_mb"] for r in ok]
    summary = {
        "total_cases": len(cases),
        "completed": len(ok),
        "failed": len(failed),
        "val_loss_mean": statistics.mean(losses) if losses else None,
        "val_loss_stdev": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
        "artifact_size_mb_mean": statistics.mean(sizes) if sizes else None,
        "artifact_size_mb_stdev": statistics.pstdev(sizes) if len(sizes) > 1 else 0.0,
        "all_results": sorted(ok, key=lambda r: r["run_name"]) + failed,
    }
    out = root / "runs" / "seed_check_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
