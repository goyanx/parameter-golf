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

    grid = [
        ("retune_base", 240, 2.2),
        ("retune_t200_temp20", 200, 2.0),
        ("retune_t200_temp24", 200, 2.4),
        ("retune_t240_temp20", 240, 2.0),
        ("retune_t280_temp20", 280, 2.0),
        ("retune_t280_temp24", 280, 2.4),
    ]

    cases: list[tuple[str, dict]] = []
    for name, teacher_steps, temp in grid:
        cfg = copy.deepcopy(base)
        cfg["distill"]["teacher_steps"] = int(teacher_steps)
        cfg["distill"]["temperature"] = float(temp)
        cases.append((name, cfg))

    print(f"Running {len(cases)} retune cases with max_workers={max_workers}")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, root, python_exe, name, cfg) for name, cfg in cases]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r.get("status") == "ok":
                print(
                    f"{r['run_name']}: size_mb={r['artifact_size_mb']:.4f} "
                    f"val_loss={r['val_loss']:.4f} t={r['teacher_steps']} temp={r['distill_temperature']}"
                )
            else:
                print(f"{r.get('run_name')}: failed")

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]
    by_loss = sorted(ok, key=lambda r: (r["val_loss"], r["artifact_size_mb"]))
    base_m = next((r for r in ok if r["run_name"] == "retune_base"), None)
    deltas = []
    if base_m is not None:
        for r in ok:
            if r["run_name"] == "retune_base":
                continue
            deltas.append(
                {
                    "run_name": r["run_name"],
                    "delta_val_loss": r["val_loss"] - base_m["val_loss"],
                    "delta_artifact_size_mb": r["artifact_size_mb"] - base_m["artifact_size_mb"],
                    "teacher_steps": r.get("teacher_steps"),
                    "temperature": r.get("distill_temperature"),
                }
            )
        deltas.sort(key=lambda d: (d["delta_val_loss"], d["delta_artifact_size_mb"]))

    summary = {
        "total_cases": len(cases),
        "completed": len(ok),
        "failed": len(failed),
        "baseline": base_m,
        "best_loss": by_loss[0] if by_loss else None,
        "deltas_vs_base": deltas,
        "all_results": sorted(ok, key=lambda r: r["run_name"]) + failed,
    }
    out = root / "runs" / "teacher_retune_probe_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
