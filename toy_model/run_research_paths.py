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


def run_one(root: Path, python_exe: str, name: str, cfg: dict, group: str) -> dict:
    tmp_cfg = root / f"tmp_{name}.yaml"
    log_dir = root / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    write_yaml(tmp_cfg, cfg)
    cmd = [python_exe, str(root / "train.py"), "--config", str(tmp_cfg), "--run-name", name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")

    if proc.returncode != 0:
        return {"run_name": name, "group": group, "status": "failed", "returncode": proc.returncode, "log": str(log_path)}

    metrics_path = root / "runs" / f"{name}.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["status"] = "ok"
    metrics["group"] = group
    metrics["log"] = str(log_path)
    return metrics


def build_cases(base_cfg: dict) -> list[tuple[str, str, dict]]:
    cases: list[tuple[str, str, dict]] = []

    def add(group: str, name: str, edits: list[tuple[tuple[str, ...], object]]) -> None:
        cfg = copy.deepcopy(base_cfg)
        for path, value in edits:
            set_nested(cfg, path, value)
        cases.append((group, name, cfg))

    # Structured sparsity path.
    add("structured", "struct_mag_011", [])
    add(
        "structured",
        "struct_row_attn_010",
        [
            (("prune", "mode"), "row"),
            (("prune", "amount"), 0.10),
            (("prune", "include_patterns"), ["attn"]),
        ],
    )
    add(
        "structured",
        "struct_row_attn_mlp_011",
        [
            (("prune", "mode"), "row"),
            (("prune", "amount"), 0.11),
            (("prune", "include_patterns"), ["attn", "mlp"]),
        ],
    )
    add(
        "structured",
        "struct_col_attn_mlp_011",
        [
            (("prune", "mode"), "col"),
            (("prune", "amount"), 0.11),
            (("prune", "include_patterns"), ["attn", "mlp"]),
        ],
    )

    # Selective precision path.
    add("precision", "prec_base", [])
    add(
        "precision",
        "prec_keep_proj_fp16",
        [
            (("quantize", "exclude_patterns"), ["token_emb", "head", "ln", "norm", "attn.proj", "mlp.fc2"]),
            (("quantize", "fallback_dtype"), "fp16"),
        ],
    )
    add(
        "precision",
        "prec_keep_proj_fp32",
        [
            (("quantize", "exclude_patterns"), ["token_emb", "head", "ln", "norm", "attn.proj", "mlp.fc2"]),
            (("quantize", "fallback_dtype"), "fp32"),
        ],
    )
    add(
        "precision",
        "prec_keep_attn_mlp_fp16",
        [
            (("quantize", "exclude_patterns"), ["token_emb", "head", "ln", "norm", "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]),
            (("quantize", "fallback_dtype"), "fp16"),
        ],
    )

    # Selective low-rank path.
    add("lowrank", "lr_all_56", [])
    add(
        "lowrank",
        "lr_mlp_56",
        [
            (("lowrank", "include_patterns"), ["mlp"]),
            (("lowrank", "exclude_patterns"), ["head", "token_emb"]),
        ],
    )
    add(
        "lowrank",
        "lr_attn_56",
        [
            (("lowrank", "include_patterns"), ["attn"]),
            (("lowrank", "exclude_patterns"), ["head", "token_emb"]),
        ],
    )
    add(
        "lowrank",
        "lr_mlp_48",
        [
            (("lowrank", "rank"), 48),
            (("lowrank", "include_patterns"), ["mlp"]),
            (("lowrank", "exclude_patterns"), ["head", "token_emb"]),
        ],
    )

    return cases


def main() -> None:
    root = Path(__file__).resolve().parent
    python_exe = os.environ.get("PYTHON_EXE", sys.executable)
    base_cfg = yaml.safe_load((root / "config_micro_best.yaml").read_text(encoding="utf-8"))
    cases = build_cases(base_cfg)

    max_workers = max(1, int(os.environ.get("MAX_WORKERS", "2")))
    results: list[dict] = []
    print(f"Running {len(cases)} cases with max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(run_one, root, python_exe, name, cfg, group)
            for group, name, cfg in cases
        ]
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            rn = res.get("run_name", "unknown")
            st = res.get("status", "unknown")
            if st == "ok":
                print(
                    f"{rn}: ok group={res['group']} size_mb={res['artifact_size_mb']:.4f} "
                    f"val_loss={res['val_loss']:.4f} time={res['train_time_sec']:.2f}s"
                )
            else:
                print(f"{rn}: failed group={res.get('group')} (see {res.get('log')})")

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]

    by_group: dict[str, list[dict]] = {}
    for r in ok:
        by_group.setdefault(str(r["group"]), []).append(r)

    summary = {
        "total_cases": len(cases),
        "completed": len(ok),
        "failed": len(failed),
        "by_group_best": {
            group: sorted(items, key=lambda r: (r["val_loss"], r["artifact_size_mb"]))[0]
            for group, items in by_group.items()
        },
        "all_results": sorted(ok, key=lambda r: r["run_name"]) + failed,
    }
    out = root / "runs" / "research_paths_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
