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


def run_one(root: Path, python_exe: str, name: str, cfg: dict, preset: str, seed: int) -> dict:
    tmp_cfg = root / f"tmp_{name}.yaml"
    log_dir = root / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    write_yaml(tmp_cfg, cfg)
    cmd = [python_exe, str(root / "train.py"), "--config", str(tmp_cfg), "--run-name", name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        return {
            "run_name": name,
            "preset": preset,
            "seed": seed,
            "status": "failed",
            "returncode": proc.returncode,
            "log": str(log_path),
        }
    out = root / "runs" / f"{name}.json"
    m = json.loads(out.read_text(encoding="utf-8"))
    m["preset"] = preset
    m["seed"] = seed
    m["status"] = "ok"
    m["log"] = str(log_path)
    return m


def _stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    losses = [r["val_loss"] for r in rows]
    sizes = [r["artifact_size_mb"] for r in rows]
    return {
        "count": len(rows),
        "val_loss_mean": statistics.mean(losses),
        "val_loss_stdev": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
        "artifact_size_mb_mean": statistics.mean(sizes),
        "artifact_size_mb_stdev": statistics.pstdev(sizes) if len(sizes) > 1 else 0.0,
        "best_loss": min(rows, key=lambda r: (r["val_loss"], r["artifact_size_mb"])),
        "best_size": min(rows, key=lambda r: (r["artifact_size_mb"], r["val_loss"])),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    python_exe = os.environ.get("PYTHON_EXE", sys.executable)
    max_workers = max(1, int(os.environ.get("MAX_WORKERS", "2")))
    seeds = [1337, 2027, 3141, 4242]

    preset_cfgs = {
        "quality": yaml.safe_load((root / "config_toyfocus_best.yaml").read_text(encoding="utf-8")),
        "compact": yaml.safe_load((root / "config_toyfocus_mixedbit_compact.yaml").read_text(encoding="utf-8")),
    }

    cases: list[tuple[str, dict, str, int]] = []
    for preset, base_cfg in preset_cfgs.items():
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["train"]["seed"] = int(seed)
            run_name = f"seed4_{preset}_{seed}"
            cases.append((run_name, cfg, preset, seed))

    print(f"Running {len(cases)} runs across 2 presets with max_workers={max_workers}")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(run_one, root, python_exe, name, cfg, preset, seed)
            for name, cfg, preset, seed in cases
        ]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r.get("status") == "ok":
                print(
                    f"{r['run_name']}: preset={r['preset']} seed={r['seed']} "
                    f"size_mb={r['artifact_size_mb']:.4f} val_loss={r['val_loss']:.4f}"
                )
            else:
                print(f"{r.get('run_name')}: failed")

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]
    by_preset: dict[str, list[dict]] = {"quality": [], "compact": []}
    for r in ok:
        by_preset.setdefault(str(r["preset"]), []).append(r)

    stats = {k: _stats(v) for k, v in by_preset.items()}
    recommendation = "quality"
    if stats.get("quality") and stats.get("compact"):
        q = stats["quality"]
        c = stats["compact"]
        # Quality-first tie-breaker with simple variance guard.
        if c["val_loss_mean"] <= q["val_loss_mean"] + 0.01 and c["artifact_size_mb_mean"] < q["artifact_size_mb_mean"]:
            recommendation = "compact"
        else:
            recommendation = "quality"

    summary = {
        "total_cases": len(cases),
        "completed": len(ok),
        "failed": len(failed),
        "seeds": seeds,
        "stats_by_preset": stats,
        "recommendation": recommendation,
        "all_results": sorted(ok, key=lambda r: (r["preset"], r["seed"])) + failed,
    }
    out = root / "runs" / "dual_preset_seed_check_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")
    print(f"Recommended preset: {recommendation}")


if __name__ == "__main__":
    main()
