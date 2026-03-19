# Memory Log

Repository memory for experiments, outcomes, and lessons learned.

## How To Use

- Append entries, never replace prior history.
- One experiment or incident per entry.
- Include metrics whenever available.

## Entry Template

```md
## YYYY-MM-DD - <short title>

Context:
- What was attempted and why.

Changes:
- Files changed:
- Config knobs:

Results:
- parameter_count:
- artifact_size_mb:
- train_time_sec:
- val_loss:
- global_sparsity:

Decision:
- Keep / Revert / Needs follow-up.

Learnings:
- Key takeaway(s).

Next:
- Immediate next action.
```

## Entries

## 2026-03-19 - Toy scaffold created

Context:
- Set up local-first playground for Parameter Golf iteration.

Changes:
- Files changed: `toy_model/*` baseline scaffold and docs.
- Config knobs: baseline config with optional quant/prune/low-rank/weight-sharing switches.

Results:
- Runtime smoke test blocked by local PyTorch DLL initialization error (`WinError 1114`, `c10.dll`).

Decision:
- Keep scaffold, fix environment next.

Learnings:
- Current environment cannot run torch reliably; experimentation depends on venv/runtime repair.

Next:
- Repair Python/PyTorch env, then run `toy_model/train.py` and `toy_model/run_ablations.py`.

## 2026-03-19 - Compression stack implemented and measured

Context:
- Added recommended compression path: group-wise INT4 + bit-packing, mixed-precision exclusions, and short QAT recovery.

Changes:
- Files changed: `toy_model/quantize.py`, `toy_model/size_report.py`, `toy_model/train.py`, `toy_model/config.yaml`, `toy_model/config_compressed.yaml`, `toy_model/run_ablations.py`, `toy_model/README.md`.
- Config knobs:
  - `quantize.bits=4`
  - `quantize.group_size=64`
  - `quantize.exclude_patterns=[token_emb, head, ln, norm]`
  - `qat.enabled=true`, `qat.steps=40`
  - `model.weight_sharing=true`
  - `lowrank.rank=32`
  - `prune.amount=0.2`

Results:
- Baseline (`baseline_new2`):
  - parameter_count: 437760
  - artifact_size_mb: 1.5539
  - train_time_sec: 5.1632
  - val_loss: 2.0207
- Compressed (`compressed_new2`):
  - parameter_count: 120704
  - artifact_size_mb: 0.1038
  - train_time_sec: 10.7644
  - val_loss: 3.0508
  - global_sparsity: 0.0467

Decision:
- Keep compression stack; quality regressed significantly under current aggressive settings.

Learnings:
- Compression is effective for artifact size (about 15x smaller vs baseline) but current config over-compresses quality.
- QAT-lite helped keep pipeline coherent but did not recover enough loss with this tiny toy setup.

Next:
- Tune less aggressive config:
  - disable low-rank first
  - lower prune amount
  - increase QAT steps

## 2026-03-19 - Distillation + QAT integrated

Context:
- Added teacher-student distillation to recover quality after aggressive compression.

Changes:
- Files changed: `toy_model/train.py`, `toy_model/config.yaml`, `toy_model/config_distill_qat.yaml`, `toy_model/README.md`.
- Config knobs:
  - `distill.enabled=true`
  - `distill.alpha=0.35`
  - `distill.temperature=2.5`
  - `distill.teacher_steps=160`
  - `qat.enabled=true`
  - `qat.steps=60`
  - `quantize.bits=4`

Results:
- Distill+QAT (`distill_qat_new`):
  - parameter_count: 239488
  - artifact_size_mb: 0.2385
  - train_time_sec: 21.7373
  - val_loss: 2.5958
- Prior compressed (`compressed_new2`):
  - parameter_count: 120704
  - artifact_size_mb: 0.1038
  - val_loss: 3.0508

Decision:
- Keep distillation path; quality improved materially over compressed-only setup.

Learnings:
- Distillation recovered a significant share of quality at still-small artifact size.
- Current distill setup increases runtime and model size due to less aggressive structural compression.

Next:
- Try distillation with lower parameter budget (re-enable low-rank cautiously) and retune alpha/temperature.

## 2026-03-19 - Tight-size distill+QAT sweep

Context:
- Goal: approach artifact ~0.12 MB while keeping val_loss < 2.8.

Changes:
- Added `toy_model/run_target_sweep.py` and executed four configs (`target_a`..`target_d`).
- Added selected tight-size preset `toy_model/config_pareto_tight.yaml`.

Results:
- `target_a`: artifact_size_mb=0.1280, val_loss=3.0292
- `target_b`: artifact_size_mb=0.1453, val_loss=3.0183
- `target_c`: artifact_size_mb=0.1252, val_loss=3.0209
- `target_d`: artifact_size_mb=0.1488, val_loss=3.0235
- Best near-size pick: `target_c` (closest to 0.12 MB with strongest size tradeoff).
- No run met strict target (`artifact ~0.12 MB` and `val_loss < 2.8`) in this sweep.

Decision:
- Keep `config_pareto_tight.yaml` as the compact Pareto candidate.

Learnings:
- Distillation substantially helps compared to compressed-only baseline, but current toy constraints still sit near val_loss~3.0 at ~0.12-0.15 MB.

Next:
- Try a larger teacher and longer teacher training while keeping student size fixed.
- Explore distillation alpha/temperature grid with no low-rank to test whether low-rank is the main quality limiter.

## 2026-03-19 - Parallel light sweep (desktop-safe)

Context:
- User requested multi-threaded experimentation with lower desktop load.

Changes:
- Added `toy_model/run_parallel_sweep.py` with configurable workers and modes.
- Executed `--max-workers 2 --mode light` (8 runs).

Results:
- Best loss run: `par_04` with val_loss=2.4548, artifact_size_mb=0.2385.
- All 8 runs completed; no failures.
- None reached ~0.12 MB in this no-lowrank light sweep (all around ~0.238 MB).

Decision:
- Keep `run_parallel_sweep.py` as the standard parallel search entrypoint.

Learnings:
- Distillation + longer QAT consistently improves quality.
- Hitting ~0.12 MB likely requires reintroducing structural compression (e.g., low-rank/stronger pruning) with better recovery.

Next:
- Run mixed sweep that includes mild low-rank and stronger distillation to recover size-quality balance.

## 2026-03-19 - Stability controls from race-car analogy

Context:
- Implemented stabilization actions: progressive pruning, distillation alpha schedule, and gradient clipping.

Changes:
- `toy_model/train.py`: added
  - linear distillation alpha schedule (`alpha_start -> alpha_end`)
  - progressive pruning (`start_step/end_step/every`)
  - gradient clipping for main training and QAT
- `toy_model/config_stability.yaml`: stability-focused preset.
- `toy_model/run_parallel_sweep.py`: mixed mode now enables stability controls.

Results:
- Single stability run (`stability_single`):
  - artifact_size_mb: 0.1354
  - val_loss: 2.9566
- Mixed sweep with same 2-worker setting (`--mode mixed --max-workers 2`):
  - Best quality: `par_02` => artifact_size_mb=0.1601, val_loss=2.6627
  - Second best: `par_04` => artifact_size_mb=0.1600, val_loss=2.6958
  - Best size in sweep: ~0.1349 MB with val_loss ~2.9535

Decision:
- Keep stability controls; they clearly improved quality and produced runs below 2.8 loss.

Learnings:
- Under tighter size (~0.135 MB), loss remains around ~2.95.
- Relaxing size slightly to ~0.16 MB crosses desired quality threshold (<2.8).

Next:
- Narrow search between 0.145-0.160 MB to chase <2.8 while minimizing size.

## 2026-03-20 - Chassis tuning sweep (group size + delayed fake quant)

Context:
- Implemented requested chassis tuning knobs from literature:
  - delayed fake quant in QAT
  - group-size sweep for INT4 quantization

Changes:
- `toy_model/train.py`: added `qat.delay_fake_quant_steps`.
- `toy_model/run_parallel_sweep.py`: added `mode=chassis` with:
  - `quant_group_size in {32, 48, 64}`
  - `qat.delay_fake_quant_steps in {0, 20}`
- Executed with `--max-workers 2`.

Results:
- 12/12 runs succeeded.
- Best loss in this sweep:
  - `par_08`: artifact_size_mb=0.1438, val_loss=2.9486, group_size=32, delay=20
- Closest size in sweep:
  - `par_11`: artifact_size_mb=0.1354, val_loss=2.9564, group_size=64, delay=0

Decision:
- Keep delayed fake quant and group-size sweep support in pipeline.

Learnings:
- Smaller group size (32) improved quality vs larger group sizes, but increased artifact size.
- Delayed fake quant (20) helped quality slightly across matching settings.
- This sweep improved quality in the 0.14 MB band but did not yet break 2.9.

Next:
- Focus on `group_size=32` and tune prune/teacher strength to push below `2.9` near `~0.14-0.15 MB`.

## 2026-03-20 - Frontier sweep (additional ideas tried)

Context:
- User requested trying anything else remaining.
- Ran targeted frontier search with:
  - distillation temperature sweep (`2.0`, `2.5`)
  - prune target (`0.10`, `0.12`)
  - low-rank (`56`, `64`)
  - group size (`32`, `48`)
  - delayed fake quant (`20`)

Changes:
- `toy_model/run_parallel_sweep.py`: added `mode=frontier`.
- Added locked presets:
  - `toy_model/config_frontier_quality.yaml`
  - `toy_model/config_frontier_compact.yaml`

Results:
- Frontier sweep best quality (`par_01`): artifact_size_mb=0.1615, val_loss=2.4930
- Frontier sweep best compact under 2.8 (`par_12`):
  - artifact_size_mb=0.1565
  - val_loss=2.6920
- Repro check (`frontier_compact_confirm`) matched:
  - artifact_size_mb=0.15645
  - val_loss=2.69196

Decision:
- Keep both frontier presets as new best-known operating points.

Learnings:
- Distillation temperature `2.0` strongly improved quality (often ~2.49) at slightly higher size.
- Temperature `2.5` gave weaker quality but better compact frontier around ~0.156 MB.

Next:
- Final narrowing:
  - hold `rank=56`, `delay=20`, `group=48`
  - micro-sweep `temp in {2.2, 2.3, 2.4}` and `prune in {0.11, 0.12}` for sub-0.156 MB with <2.8.

## 2026-03-20 - Promotion to main train_gpt pipeline

Context:
- Promoted stable toy-model compression controls into `train_gpt.py` behind env flags.

Changes:
- Added optional progressive pruning controls.
- Added optional int8 fake-quant-in-training controls with delayed start.
- Added compression knob logging and final global sparsity logging.
- Preserved default behavior (all new knobs default to off/no-op).

Validation:
- `python -m py_compile train_gpt.py` passed.

Next:
- Run a short real-data smoke run with `QAT_INT8_ENABLED=1` and `PRUNE_PROGRESSIVE=1` to calibrate wall-clock and val_bpb impact.
