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


def set_nested(cfg: dict, path: tuple[str, ...], value) -> None:
    node = cfg
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


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

    metrics_path = root / "runs" / f"{name}.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["status"] = "ok"
    metrics["log"] = str(log_path)
    return metrics


def build_cases(base_cfg: dict) -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []

    def add(name: str, prune_amt: float) -> None:
        cfg = copy.deepcopy(base_cfg)
        set_nested(cfg, ("model", "attn_kv_heads"), 2)
        set_nested(cfg, ("quantize", "pack_order"), "size_desc")
        set_nested(cfg, ("prune", "mode"), "row")
        set_nested(cfg, ("prune", "amount"), prune_amt)
        set_nested(cfg, ("prune", "include_patterns"), ["attn"])
        cases.append((name, cfg))

    for prune_amt in [0.065, 0.075, 0.08, 0.085, 0.095]:
        add(f"arch_combo_tight_{str(prune_amt).replace('.', '_')}", prune_amt)
    return cases


def main() -> None:
    root = Path(__file__).resolve().parent
    python_exe = os.environ.get("PYTHON_EXE", sys.executable)
    base_cfg = yaml.safe_load((root / "config_arch_combo_best.yaml").read_text(encoding="utf-8"))
    cases = build_cases(base_cfg)

    max_workers = max(1, int(os.environ.get("MAX_WORKERS", "2")))
    results: list[dict] = []
    print(f"Running {len(cases)} cases with max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, root, python_exe, name, cfg) for name, cfg in cases]
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            rn = res.get("run_name", "unknown")
            st = res.get("status", "unknown")
            if st == "ok":
                print(
                    f"{rn}: ok size_mb={res['artifact_size_mb']:.4f} "
                    f"val_loss={res['val_loss']:.4f} time={res['train_time_sec']:.2f}s"
                )
            else:
                print(f"{rn}: failed (see {res.get('log')})")

    ok = [r for r in results if r.get("status") == "ok"]
    summary = {
        "best_loss": sorted(ok, key=lambda r: (r["val_loss"], r["artifact_size_mb"]))[0] if ok else None,
        "best_size": sorted(ok, key=lambda r: (r["artifact_size_mb"], r["val_loss"]))[0] if ok else None,
        "all_results": sorted(ok, key=lambda r: r["run_name"]),
    }
    out = root / "runs" / "arch_combo_tight_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
