# Toy Parameter Golf Workspace

Local-first playground for compression-centric language model experiments.

## Goals

- Validate train/eval loop quickly on small data.
- Measure artifact size and fail if it exceeds 16 MB.
- Run ablations for quantization, pruning, weight sharing, and low-rank layers.

## Project Layout

```
toy_model/
  config.yaml
  model.py
  train.py
  quantize.py
  prune.py
  lowrank.py
  tokenizer.py
  size_report.py
  run_ablations.py
  data/tiny_corpus.txt
  runs/
```

## Quick Start

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch pyyaml
python toy_model\train.py --config toy_model\config.yaml --run-name baseline
python toy_model\train.py --config toy_model\config_compressed.yaml --run-name compressed
python toy_model\train.py --config toy_model\config_distill_qat.yaml --run-name distill_qat
```

Expected output includes:

- `parameter_count`
- `artifact_size_mb`
- `train_time_sec`
- `val_loss`

Metrics are written to `toy_model/runs/<run-name>.json`.

## Run Ablations

```powershell
python toy_model\run_ablations.py
```

This runs:

- `baseline`
- `quant4_qat`
- `prune30`
- `weight_share`
- `lowrank32`

Summary saved to `toy_model/runs/ablation_summary.json`.

## Research Paths

```powershell
python toy_model\run_research_paths.py
```

This runs three parallel mini-sweeps from `config_micro_best.yaml`:

- structured sparsity
- selective precision
- selective low-rank

Summary saved to `toy_model/runs/research_paths_summary.json`.

```powershell
python toy_model\run_breadth_sweep.py
```

This runs a broader parallel sweep over:

- structured pruning on attention and MLP blocks
- selective low-rank on attention and MLP blocks
- artifact packing order

Summary saved to `toy_model/runs/breadth_sweep_summary.json`.

```powershell
python toy_model\run_pack_compare.py
```

This rechecks the current structured best with different packing orders:

- `state_dict`
- `name`
- `size_desc`

Summary saved to `toy_model/runs/pack_compare_summary.json`.

```powershell
python toy_model\run_arch_sweep.py
```

This runs a small architecture sweep over grouped-query attention:

- `model.attn_kv_heads=4` baseline
- `model.attn_kv_heads=2`
- `model.attn_kv_heads=1`

Summary saved to `toy_model/runs/arch_sweep_summary.json`.

```powershell
python toy_model\run_arch_combo.py
```

This combines the best architecture knob with the best structured-pruning recipe:

- `model.attn_kv_heads=2`
- `prune.mode=row`
- `prune.include_patterns=["attn"]`
- `prune.amount` swept around the compact frontier

Summary saved to `toy_model/runs/arch_combo_summary.json`.

## Config Knobs

- `quantize.bits`: `null` or integer in `[2..8]`
- `quantize.group_size`: group size for group-wise quantization
- `quantize.exclude_patterns`: parameter-name substrings kept at fallback precision
- `quantize.fallback_dtype`: `fp16` or `fp32` for excluded tensors
- `quantize.pack_order`: `state_dict`, `name`, or `size_desc` ordering for payload packing
- `prune.amount`: fraction in `[0.0, 1.0)`
- `prune.mode`: `magnitude`, `row`, or `col`
- `prune.include_patterns`: optional substrings that restrict pruning to selected layers
- `prune.exclude_patterns`: optional substrings that skip selected layers
- `model.weight_sharing`: `true|false`
- `model.attn_kv_heads`: grouped-query attention KV head count; defaults to `n_heads`
- `lowrank.rank`: `null` or positive integer
- `lowrank.include_patterns`: optional substrings that restrict low-rank replacement
- `lowrank.exclude_patterns`: optional substrings that skip low-rank replacement
- `qat.enabled|steps|lr`: short quantization-recovery finetune after main training
- `distill.enabled|alpha|temperature`: enable teacher-student distillation
- `distill.teacher_steps|teacher_lr|teacher_model`: teacher training configuration

## Notes

- This is a toy loop and not the official challenge training setup.
- Artifact sizing uses compressed packaging of code files plus serialized model payload estimate.
- `train.py` hard-fails if estimated artifact size exceeds `16 MB`.
