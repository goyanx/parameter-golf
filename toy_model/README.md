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

## Config Knobs

- `quantize.bits`: `null` or integer in `[2..8]`
- `quantize.group_size`: group size for group-wise quantization
- `quantize.exclude_patterns`: parameter-name substrings kept at fallback precision
- `quantize.fallback_dtype`: `fp16` or `fp32` for excluded tensors
- `prune.amount`: fraction in `[0.0, 1.0)`
- `model.weight_sharing`: `true|false`
- `lowrank.rank`: `null` or positive integer
- `qat.enabled|steps|lr`: short quantization-recovery finetune after main training
- `distill.enabled|alpha|temperature`: enable teacher-student distillation
- `distill.teacher_steps|teacher_lr|teacher_model`: teacher training configuration

## Notes

- This is a toy loop and not the official challenge training setup.
- Artifact sizing uses compressed packaging of code files plus serialized model payload estimate.
- `train.py` hard-fails if estimated artifact size exceeds `16 MB`.
