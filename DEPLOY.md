# Deploy Notes (Runpod)

Reference notes for deploying `train_gpt.py` experiments with promoted compression controls.

## Goal

- Run real-pipeline calibration and tuning on Runpod without changing default baseline behavior.
- Use env flags to enable/disable compression features safely.

## Prerequisites

- Dataset shards exist and match expected format:
  - `fineweb_train_*.bin`
  - `fineweb_val_*.bin`
- Tokenizer exists:
  - `fineweb_1024_bpe.model`
- CUDA environment is healthy.

## Core Paths

- Script: `train_gpt.py`
- Main knobs source: `Hyperparameters` env vars in `train_gpt.py`

## Promoted Compression Knobs

### Progressive pruning

- `PRUNE_AMOUNT` (default `0.0`)
- `PRUNE_PROGRESSIVE` (`0|1`, default `0`)
- `PRUNE_START_STEP` (default auto: half of iterations)
- `PRUNE_END_STEP` (default auto: `ITERATIONS`)
- `PRUNE_EVERY` (default `100`)
- `PRUNE_EXCLUDE` (default `tok_emb,lm_head,final_norm,skip_weights`)

### Int8 fake-quant in training

- `QAT_INT8_ENABLED` (`0|1`, default `0`)
- `QAT_INT8_START_STEP` (default auto: half of iterations)
- `QAT_INT8_EVERY` (default `1`)
- `QAT_INT8_EXCLUDE` (default `tok_emb,lm_head,final_norm,skip_weights`)

## Recommended Run Sequence

1. Baseline sanity run (all new knobs off).
2. Pruning-only run.
3. QAT-int8-only run.
4. Combined run (pruning + QAT-int8).
5. Compare `val_bpb`, `val_loss`, wall-clock, and final artifact size.

## Example Env Presets

### A) Baseline (control)

```powershell
$env:PRUNE_AMOUNT="0.0"
$env:PRUNE_PROGRESSIVE="0"
$env:QAT_INT8_ENABLED="0"
python train_gpt.py
```

### B) Stability/compression candidate

```powershell
$env:PRUNE_AMOUNT="0.12"
$env:PRUNE_PROGRESSIVE="1"
$env:PRUNE_START_STEP="10000"
$env:PRUNE_END_STEP="20000"
$env:PRUNE_EVERY="200"
$env:QAT_INT8_ENABLED="1"
$env:QAT_INT8_START_STEP="12000"
$env:QAT_INT8_EVERY="1"
python train_gpt.py
```

### C) Gentler candidate

```powershell
$env:PRUNE_AMOUNT="0.10"
$env:PRUNE_PROGRESSIVE="1"
$env:PRUNE_START_STEP="11000"
$env:PRUNE_END_STEP="20000"
$env:PRUNE_EVERY="250"
$env:QAT_INT8_ENABLED="1"
$env:QAT_INT8_START_STEP="13000"
$env:QAT_INT8_EVERY="1"
python train_gpt.py
```

## What To Verify In Logs

- `compression_knobs:` line is printed with expected values.
- Validation lines:
  - `val_loss`
  - `val_bpb`
- Early stop behavior:
  - `stopping_early: wallclock_cap ...` (if max wallclock is hit)
- Serialization lines:
  - `Serialized model int8+zlib: ...`
  - `Total submission size int8+zlib: ...`
- Final sparsity line:
  - `model_sparsity: nonzero:... total:... global_sparsity:...`

## Runpod Checklist

- Confirm mounted storage path for dataset and tokenizer.
- Export `DATA_PATH` and `TOKENIZER_PATH` correctly.
- Use fixed `RUN_ID` per trial.
- Capture stdout/stderr to a run-specific log file.
- Persist generated artifacts:
  - `final_model.pt`
  - `final_model.int8.ptz`
  - log file

## Suggested First Runpod Matrix

- 3-run matrix:
  1. Baseline
  2. `PRUNE_AMOUNT=0.10` + progressive + QAT-int8
  3. `PRUNE_AMOUNT=0.12` + progressive + QAT-int8

Keep all other hyperparameters fixed for fair comparison.

## Smoke A/B Helper

Use this helper to compare baseline vs `COMPRESSION_PRESET=toy_micro_best` quickly:

```powershell
python tools\run_main_smoke_compare.py --max-wallclock-seconds 120 --iterations 20000 --run-id-prefix runpod_smoke
```

Outputs:

- logs under `logs/main_smoke_compare_<timestamp>/`
- parsed summary JSON at `logs/main_smoke_compare_<timestamp>/summary.json`

## Notes

- Defaults remain unchanged when new env vars are unset.
- These knobs are calibration-stage controls; final record submissions should still prioritize reproducibility and size/bpb tradeoff validation.
