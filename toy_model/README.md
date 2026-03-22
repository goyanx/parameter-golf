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
  run_parallel_sweep.py
  run_ablations.py
  run_target_sweep.py
  run_structured_tight.py
  run_precision_ranked.py
  run_breadth_sweep.py
  run_pack_compare.py
  run_arch_sweep.py
  run_arch_combo.py
  run_hidden_distill_probe.py
  run_alibi_probe.py
  run_mixedbit_probe.py
  run_attn_distill_probe.py
  run_sparse24_pack_probe.py
  run_teacher_retune_probe.py
  run_toyfocus_confirm_probe.py
  run_mtp_probe.py
  run_seed_check.py
  run_dual_preset_seed_check.py
  run_research_paths.py
  EXPERIMENT_TRACKER.md
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

## Current Best Toy Configs

- Best compact-quality in recent toy sweeps:
  - `toy_model/config_toyfocus_best.yaml`
  - uses `model.positional_encoding=alibi`
  - uses light attention-map distillation (`distill.attn_enabled=true`, `attn_weight=0.02`)
  - confirmed with `teacher_steps=300`, `temperature=2.0`
  - uses MTP aux loss (`mtp.enabled=true`, `horizons=[2,3]`, `weight=0.1`)
- Best quality-focused toy preset:
  - `toy_model/config_toyfocus_quality_best.yaml`
- Best compact-size variant with small quality tradeoff:
  - `toy_model/config_toyfocus_mixedbit_compact.yaml`
- Strong compact baseline:
  - `toy_model/config_micro_best.yaml`
- Frontier quality-oriented:
  - `toy_model/config_frontier_quality.yaml`
- Frontier compact-oriented:
  - `toy_model/config_frontier_compact.yaml`

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

## Main Sweep Runner

Use the parallel sweep runner for most iteration:

```powershell
python toy_model\run_parallel_sweep.py --max-workers 2 --mode <mode>
```

Supported modes:

- `light`: quick starter sweep
- `mixed`: mild structural compression + recovery
- `chassis`: quant group-size + delayed fake quant sweep
- `frontier`: broader last-mile sweep
- `micro`: tight local sweep around compact frontier
- `refine`: compact frontier refinement with group-size midpoint
- `tight`: final squeeze around current best compact point
- `toyfocus`: grouped-query architecture focus around compact settings
- `full`: larger search (heavier)

Primary output:

- `toy_model/runs/parallel_sweep_summary.json`

Per-run logs:

- `toy_model/runs/logs/par_XX.log`

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

## Recommended Workflow (Toy-Only)

1. Start from a locked config:
   - `config_toyfocus_best.yaml` or `config_micro_best.yaml`
2. Run one confirmation:
   - `python toy_model\train.py --config toy_model\config_toyfocus_best.yaml --run-name confirm`
3. Run a narrow 2-worker sweep:
   - `python toy_model\run_parallel_sweep.py --max-workers 2 --mode toyfocus`
4. Review summary:
   - `toy_model/runs/parallel_sweep_summary.json`
5. Log and decide:
   - update `toy_model/EXPERIMENT_TRACKER.md`
   - append key results to top-level `memory.md`

## Config Knobs

- `quantize.bits`: `null` or integer in `[2..8]`
- `quantize.group_size`: group size for group-wise quantization
- `quantize.exclude_patterns`: parameter-name substrings kept at fallback precision
- `quantize.fallback_dtype`: `fp16` or `fp32` for excluded tensors
- `quantize.pack_order`: `state_dict`, `name`, or `size_desc` ordering for payload packing
- `quantize.layer_bits`: optional per-layer bit overrides, e.g. `{"attn.qkv": 3, "mlp.fc1": 5}`
- `quantize.sparse_2_4_pack`: enable sparse-aware payload encoding for valid 2:4-pruned tensors
- `prune.amount`: fraction in `[0.0, 1.0)`
- `prune.mode`: `magnitude`, `row`, or `col`
- `prune.mode`: `nm2_4` for deterministic 2:4 pruning on selected layers
- `prune.include_patterns`: optional substrings that restrict pruning to selected layers
- `prune.exclude_patterns`: optional substrings that skip selected layers
- `model.weight_sharing`: `true|false`
- `model.attn_kv_heads`: grouped-query attention KV head count; defaults to `n_heads`
- `model.positional_encoding`: `learned` or `alibi`
- `lowrank.rank`: `null` or positive integer
- `lowrank.include_patterns`: optional substrings that restrict low-rank replacement
- `lowrank.exclude_patterns`: optional substrings that skip low-rank replacement
- `qat.enabled|steps|lr`: short quantization-recovery finetune after main training
- `distill.enabled|alpha|temperature`: enable teacher-student distillation
- `distill.hidden_enabled|hidden_weight`: optional TinyBERT-style hidden-state distillation
- `distill.attn_enabled|attn_weight`: optional attention-map distillation loss
- `distill.teacher_steps|teacher_lr|teacher_model`: teacher training configuration
- `mtp.enabled|weight|horizons`: optional multi-token prediction auxiliary loss

### Hidden Distillation Probe

```powershell
python toy_model\run_hidden_distill_probe.py
```

Compares:

- baseline logits distillation
- logits + hidden-state distillation with small hidden-loss weights

Summary saved to `toy_model/runs/hidden_distill_probe_summary.json`.

### ALiBi Probe

```powershell
python toy_model\run_alibi_probe.py
```

Compares:

- learned positional embeddings
- ALiBi positional bias (`model.positional_encoding=alibi`)

Summary saved to `toy_model/runs/alibi_probe_summary.json`.

### Mixed-Bit Probe

```powershell
python toy_model\run_mixedbit_probe.py
```

Compares mixed-bit quantization variants on top of the ALiBi toyfocus baseline.

Summary saved to `toy_model/runs/mixedbit_probe_summary.json`.

### Attention Distillation Probe

```powershell
python toy_model\run_attn_distill_probe.py
```

Compares logits-only distillation against logits+attention-map distillation.

Summary saved to `toy_model/runs/attn_distill_probe_summary.json`.

### Sparse 2:4 Packing Probe

```powershell
python toy_model\run_sparse24_pack_probe.py
```

Compares baseline vs `nm2_4` pruning and checks sparse-aware payload packing impact.

Summary saved to `toy_model/runs/sparse24_pack_probe_summary.json`.

### Teacher Retune Probe

```powershell
python toy_model\run_teacher_retune_probe.py
```

Runs a tiny sweep on teacher strength and distillation temperature around the current best toy config.

Summary saved to `toy_model/runs/teacher_retune_probe_summary.json`.

### Toyfocus Confirmation Probe

```powershell
python toy_model\run_toyfocus_confirm_probe.py
```

Runs a tight 4-case confirmation around the current toyfocus best settings.

Summary saved to `toy_model/runs/toyfocus_confirm_probe_summary.json`.

### MTP Probe

```powershell
python toy_model\run_mtp_probe.py
```

Compares baseline against multi-token prediction (MTP) auxiliary losses.

Summary saved to `toy_model/runs/mtp_probe_summary.json`.

### Seed Check

```powershell
python toy_model\run_seed_check.py
```

Runs a quick 2-seed robustness check on `config_toyfocus_best.yaml`.

Summary saved to `toy_model/runs/seed_check_summary.json`.

### Dual Preset Seed Check

```powershell
python toy_model\run_dual_preset_seed_check.py
```

Runs 4-seed comparisons for both quality and compact presets and recommends one.

Summary saved to `toy_model/runs/dual_preset_seed_check_summary.json`.

## Notes

- This is a toy loop and not the official challenge training setup.
- Artifact sizing uses compressed packaging of code files plus serialized model payload estimate.
- `train.py` hard-fails if estimated artifact size exceeds `16 MB`.
- Toy runs primarily track `val_loss`; official challenge ranking uses `val_bpb` in `train_gpt.py`.

## Scale-Up Workflow (Local)

Build a larger local corpus slice (about 2 MB) from project docs and any cached `data/docs_selected.jsonl` content:

```powershell
python toy_model\build_local_corpus.py --out data/local_slice_corpus.txt --target-bytes 2000000
```

Run the desktop scale-up preset:

```powershell
python toy_model\train.py --config toy_model\config_scaleup_desktop.yaml --run-name scaleup_desktop
```

Run a quick guardrail comparison (baseline vs scale-up) with budget simulation:

```powershell
python toy_model\run_scaleup_guardrail.py --steps 50 --max-seconds 25
```

The guardrail summary is saved to `toy_model/runs/scaleup_guardrail_summary.json`.
