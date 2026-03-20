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
    metrics = json.loads(out.read_text(encoding="utf-8"))
    metrics["status"] = "ok"
    metrics["log"] = str(log_path)
    return metrics


def _mk(base: dict, bits: int, layer_bits: dict[str, int]) -> dict:
    cfg = copy.deepcopy(base)
    cfg["quantize"]["bits"] = bits
    cfg["quantize"]["layer_bits"] = layer_bits
    return cfg


def main() -> None:
    root = Path(__file__).resolve().parent
    python_exe = os.environ.get("PYTHON_EXE", sys.executable)
    max_workers = max(1, int(os.environ.get("MAX_WORKERS", "2")))

    base = yaml.safe_load((root / "config_toyfocus_best.yaml").read_text(encoding="utf-8"))

    cases = [
        ("mixedbit_base", _mk(base, bits=4, layer_bits={})),
        ("mixedbit_attn3", _mk(base, bits=4, layer_bits={"attn.qkv": 3, "attn.proj": 3})),
        ("mixedbit_mlp3", _mk(base, bits=4, layer_bits={"mlp.fc1": 3, "mlp.fc2": 3})),
        ("mixedbit_lowrank3", _mk(base, bits=4, layer_bits={".a": 3, ".b": 3})),
        ("mixedbit_attn5_mlp3", _mk(base, bits=4, layer_bits={"attn.qkv": 5, "attn.proj": 5, "mlp.fc1": 3, "mlp.fc2": 3})),
        ("mixedbit_all3", _mk(base, bits=3, layer_bits={})),
        ("mixedbit_all3_attn5", _mk(base, bits=3, layer_bits={"attn.qkv": 5, "attn.proj": 5})),
    ]

    print(f"Running {len(cases)} mixed-bit cases with max_workers={max_workers}")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, root, python_exe, name, cfg) for name, cfg in cases]
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            if res.get("status") == "ok":
                print(f"{res['run_name']}: size_mb={res['artifact_size_mb']:.4f} val_loss={res['val_loss']:.4f}")
            else:
                print(f"{res.get('run_name')}: failed")

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]
    by_loss = sorted(ok, key=lambda r: (r["val_loss"], r["artifact_size_mb"]))
    by_size = sorted(ok, key=lambda r: (r["artifact_size_mb"], r["val_loss"]))

    baseline = next((r for r in ok if r["run_name"] == "mixedbit_base"), None)
    compact_wins = []
    if baseline is not None:
        for r in ok:
            if r["run_name"] == "mixedbit_base":
                continue
            if r["artifact_size_mb"] <= baseline["artifact_size_mb"] and r["val_loss"] <= baseline["val_loss"]:
                compact_wins.append(r)
        compact_wins = sorted(compact_wins, key=lambda r: (r["val_loss"], r["artifact_size_mb"]))

    summary = {
        "total_cases": len(cases),
        "completed": len(ok),
        "failed": len(failed),
        "baseline": baseline,
        "best_loss": by_loss[0] if by_loss else None,
        "best_size": by_size[0] if by_size else None,
        "pareto_wins_vs_base": compact_wins,
        "all_results": sorted(ok, key=lambda r: r["run_name"]) + failed,
    }

    out = root / "runs" / "mixedbit_probe_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
