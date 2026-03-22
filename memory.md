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

## 2026-03-20 - Baseline ablations rerun and decisions logged

Context:
- Completed pending checklist ablation run to refresh keep/drop decisions with current code.

Changes:
- Executed `toy_model/run_ablations.py` end-to-end.
- Reviewed `toy_model/runs/ablation_summary.json`.

Results:
- `baseline`: `artifact_size_mb~1.5572`, `val_loss~2.0207`
- `quant4_qat`: `artifact_size_mb~0.3482`, `val_loss~2.0610`
- `prune30`: `artifact_size_mb~1.2232`, `val_loss~2.3264`
- `weight_share`: `artifact_size_mb~0.8586`, `val_loss~2.4389`
- `lowrank32`: `artifact_size_mb~0.6481`, `val_loss~3.0505`

Decision:
- Keep: `quant4_qat` as the best baseline compression move.
- Conditional: `weight_share` only if extra size pressure is needed.
- Drop for now: `prune30`, `lowrank32` in isolation.

Learnings:
- QAT-quantized baseline preserves quality best among single-step compressions.
- Naive low-rank and heavy pruning remain too lossy without stronger recovery.

Next:
- Highest-value coding items now:
  1. Add a real-pipeline smoke comparator for `train_gpt.py` baseline vs `COMPRESSION_PRESET=toy_micro_best`.
  2. Keep toy exploration focused on non-plateaued axes only (architecture + selective structured compression).

## 2026-03-20 - Toy-only grouped-query focus sweep

Context:
- RunPod credits are unavailable, so focus shifted back to toy-only experimentation.
- Tested grouped-query attention as a new non-plateaued axis on top of compact frontier settings.

Changes:
- `toy_model/run_parallel_sweep.py`: added `mode=toyfocus`.
- Added `toy_model/config_toyfocus_best.yaml`.

Results:
- Best quality in sweep (`par_04`):
  - artifact_size_mb: ~0.1575
  - val_loss: ~2.5415
  - `attn_kv_heads=1`, `prune.amount=0.12`
- Compact edge case (`par_05`):
  - artifact_size_mb: ~0.1517
  - val_loss: ~2.9000
  - `attn_kv_heads=2`, `prune.amount=0.12`

Decision:
- Keep `config_toyfocus_best.yaml` as the new toy quality/size winner.

Learnings:
- In this setup, `attn_kv_heads=1` unexpectedly improved the compact frontier relative to prior micro-best.
- `attn_kv_heads=2` opens a promising path near ~0.152 MB if we can recover a small amount of loss.

Next:
- Tighten around the compact edge (`attn_kv_heads=2`, ~0.152 MB) to try crossing below `2.8` loss.

## 2026-03-20 - Fresh parallel rerun (toyfocus vs micro)

Context:
- User requested rerunning parallel tests and reporting back.
- Ran both sweeps with the same desktop-safe settings (`--max-workers 2`).

Changes:
- Commands executed:
  - `python toy_model/run_parallel_sweep.py --max-workers 2 --mode micro`
  - `python toy_model/run_parallel_sweep.py --max-workers 2 --mode toyfocus`
- Saved snapshots:
  - `toy_model/runs/parallel_sweep_micro_latest.json`
  - `toy_model/runs/parallel_sweep_toyfocus_latest.json`

Results:
- Micro best:
  - run: `par_01`
  - artifact_size_mb: `0.161449`
  - val_loss: `2.594066`
- Toyfocus best:
  - run: `par_04`
  - artifact_size_mb: `0.158110`
  - val_loss: `2.541510`
- Toyfocus compact edge:
  - `artifact_size_mb~0.1523`
  - `val_loss~2.9000`

Decision:
- Keep `toyfocus` as the stronger current branch versus `micro`.

Learnings:
- Grouped-query tuning around the compact recipe is still the best quality/size tradeoff among these two sweep families.

Next:
- Tighten only around `toyfocus` (`attn_kv_heads=1/2`) with small prune/temp perturbations.

## 2026-03-20 - Wanda pruning probe (paper idea #1)

Context:
- Implemented and tested Wanda-style pruning (`|w| * activation_norm`) against current magnitude baseline.

Changes:
- Files changed:
  - `toy_model/prune.py`
  - `toy_model/train.py`
  - `toy_model/run_wanda_probe.py`
- Added `prune.mode=wanda` and activation-calibration batches (`prune.wanda_calib_batches`).

Results:
- Magnitude baseline (`wanda_probe_magnitude`):
  - artifact_size_mb: `0.159112`
  - val_loss: `2.541510`
- Wanda (`wanda_probe_wanda`):
  - artifact_size_mb: `0.158799`
  - val_loss: `2.587195`
- Delta (`wanda - magnitude`):
  - size: `-0.000313 MB` (slightly smaller)
  - loss: `+0.045685` (worse)

Decision:
- Drop Wanda in current form for toy frontier work.

Learnings:
- This lightweight Wanda implementation did not improve the quality/size frontier.
- The tiny size gain is not worth the observed loss regression.

Next:
- Keep using magnitude/structured modes for current branch.
- If revisiting Wanda later, limit scope to selected layers and test non-progressive one-shot pruning.

## 2026-03-20 - AWQ-lite channel protection probe (paper idea #3)

Context:
- Implemented AWQ-lite style channel-level protection in quantization/QAT path.

Changes:
- Files changed:
  - `toy_model/quantize.py`
  - `toy_model/size_report.py`
  - `toy_model/train.py`
  - `toy_model/run_awq_lite_probe.py`
- Added quantization knobs:
  - `quantize.protect_frac`
  - `quantize.protect_min_cols`

Results:
- Baseline (`protect_frac=0.00`):
  - artifact_size_mb: `0.159883`
  - val_loss: `2.541510`
- AWQ-lite (`protect_frac=0.02`):
  - artifact_size_mb: `0.168137`
  - val_loss: `2.538804`
  - delta vs base: `loss -0.002706`, `size +0.008254 MB`
- AWQ-lite (`protect_frac=0.05`):
  - artifact_size_mb: `0.169952`
  - val_loss: `2.540311`
  - delta vs base: `loss -0.001199`, `size +0.010070 MB`

Decision:
- Drop AWQ-lite for current compact frontier goals.

Learnings:
- Channel protection gives only tiny quality gains while adding non-trivial size overhead.
- Not competitive against current objective (min size with strong loss).

Next:
- Move to another research axis (e.g., intermediate-feature distillation) instead of more AWQ-lite tuning.

## 2026-03-20 - MLA latent KV bottleneck probe (DeepSeek-inspired)

Context:
- Implemented an MLA-style latent KV bottleneck in attention and tested whether it improves compact frontier.

Changes:
- Files changed:
  - `toy_model/model.py`
  - `toy_model/train.py`
  - `toy_model/run_mla_probe.py`
- New model knob:
  - `model.attn_kv_latent_dim` (`null` disables)

Results:
- Base (`attn_kv_latent_dim=null`):
  - artifact_size_mb: `0.241073`
  - val_loss: `1.535272`
- Latent 16:
  - artifact_size_mb: `0.238442` (delta `-0.002630`)
  - val_loss: `2.074323` (delta `+0.539051`)
- Latent 24:
  - artifact_size_mb: `0.239343` (delta `-0.001730`)
  - val_loss: `1.740130` (delta `+0.204858`)
- Latent 32:
  - artifact_size_mb: `0.240205` (delta `-0.000868`)
  - val_loss: `1.902933` (delta `+0.367661`)

Decision:
- Drop MLA latent KV for current toy objective.

Learnings:
- In this setup, latent KV gave only tiny size savings while causing meaningful quality regressions.
- Not competitive with current compact frontier tradeoff.

Next:
- Continue with other research axes (e.g., intermediate-feature distillation), not MLA bottleneck.

## 2026-03-20 - Three-path toy research sweep

Context:
- Wanted to test the Marie Kondo framing as concrete toy-model research paths rather than a single vague compression bucket.

Changes:
- Files changed: `toy_model/prune.py`, `toy_model/lowrank.py`, `toy_model/train.py`, `toy_model/run_research_paths.py`, `toy_model/config_structured_best.yaml`, `toy_model/README.md`.
- Config knobs:
  - `prune.mode in {magnitude,row,col}`
  - `prune.include_patterns`
  - `lowrank.include_patterns`
  - `lowrank.exclude_patterns`

Results:
- Structured sparsity path:
  - Best: `struct_row_attn_010`
  - artifact_size_mb: 0.15899658203125
  - val_loss: 2.56418770551682
- Selective precision path:
  - Best loss-focused run: `prec_keep_attn_mlp_fp16`
  - artifact_size_mb: 0.304649353027344
  - val_loss: 2.54905834794044
- Selective low-rank path:
  - Best loss-focused run: `lr_attn_56`
  - artifact_size_mb: 0.234420776367188
  - val_loss: 2.00412733852863

Decision:
- Keep structured sparsity as the most promising frontier-preserving idea.
- Keep selective precision and selective low-rank as quality probes, but they are not size-efficient in the current toy setup.

Learnings:
- Row pruning on attention blocks improved loss slightly without materially changing size.
- Extra precision buys quality quickly, but at a large artifact-size cost.
- Selective low-rank on attention layers greatly improved loss, but the size penalty was too large for the current goal.

Next:
- Use `config_structured_best.yaml` as the next compact candidate.
- Keep low-rank and selective precision only as secondary comparison paths unless size math changes.

## 2026-03-20 - Structured sparsity tightened and plateaued

Context:
- Wanted to see whether attention-row pruning could move the compact frontier after the three-path sweep.

Changes:
- Files changed: `toy_model/run_structured_tight.py`, `toy_model/EXPERIMENT_TRACKER.md`, `toy_model/config_structured_best.yaml`.
- Config knobs:
  - `prune.amount in {0.06,0.08,0.10,0.12,0.14}`
  - `prune.mode=row`
  - `prune.include_patterns=["attn"]`

Results:
- Best loss:
  - `struct_tight_03`
  - artifact_size_mb: 0.15916728973388672
  - val_loss: 2.564187705516815
- Best size:
  - `struct_tight_01`
  - artifact_size_mb: 0.15910911560058594
  - val_loss: 2.5748329758644104

Decision:
- Stop pushing structured row pruning.

Learnings:
- Pruning amount changes barely moved artifact size.
- The gain over the micro-best frontier was marginal and not worth more search budget.

Next:
- Move to a different compression axis if we continue toy-model work.

## 2026-03-20 - Layer-wise precision ranking plateau

Context:
- Tested whether keeping only the most sensitive tensors/modules in higher precision would improve the compact frontier.

Changes:
- Files changed: `toy_model/train.py`, `toy_model/run_precision_ranked.py`.
- Config knobs:
  - baseline from `config_micro_best.yaml`
  - precision keep sets derived from sensitivity ranking
  - `quantize.exclude_patterns` expanded per ranked keep set

Results:
- Baseline compact run:
  - artifact_size_mb: 0.1597
  - val_loss: 2.5763
- Best single-module sensitivity keep:
  - `pos_emb`
  - artifact_size_mb: 0.1670
  - val_loss: 2.5742
- Best two-module keep:
  - `pos_emb + shared_block.attn.qkv`
  - artifact_size_mb: 0.2061
  - val_loss: 2.5510

Decision:
- Stop this branch.

Learnings:
- The most sensitive tensors were small enough that protecting them did not move the frontier in a useful way.
- Precision allocation improved loss only by spending too many extra bytes.

Next:
- Try a different compression axis rather than more precision slicing.

## 2026-03-20 - Breadth sweep across pruning, low-rank, and packing

Context:
- Ran the next broad pass to cover the most promising remaining compression axes in parallel.

Changes:
- Files changed: `toy_model/quantize.py`, `toy_model/size_report.py`, `toy_model/train.py`, `toy_model/run_breadth_sweep.py`.
- Config knobs:
  - `prune.mode in {row,col}`
  - `prune.include_patterns=["attn"]` or `["attn","mlp"]`
  - `lowrank.include_patterns=["attn"]`, `["mlp"]`, or both
  - `quantize.pack_order in {state_dict,name,size_desc}`

Results:
- Best structured run:
  - `block_row_attn_010`
  - artifact_size_mb: 0.1594371795654297
  - val_loss: 2.564187705516815
- Best packing run:
  - `pack_size_desc`
  - artifact_size_mb: 0.1594085693359375
  - val_loss: 2.57634699344635
- Best low-rank run:
  - `lr_attn_48`
  - artifact_size_mb: 0.2318716049194336
  - val_loss: 1.9322985410690308

Decision:
- Keep structured row pruning as the only meaningful frontier-preserving branch from this sweep.
- Treat pack-order tuning as a tiny byte optimization.
- Stop low-rank as a size-efficient direction for now.

Learnings:
- Broadening pruning from attention-only to attention+MLP did not improve the frontier.
- Low-rank on attention remains quality-heavy and size-expensive.
- Sorting packed tensors by size is a real but very small artifact-size improvement.

Next:
- If we continue toy-model work, combine the best structured recipe with `quantize.pack_order=size_desc`.

## 2026-03-20 - Structured pack-order compare

Context:
- Re-ran the best structured pruning recipe with different packing orders to see whether serialization layout can beat the current structured frontier.

Changes:
- Files changed: `toy_model/run_pack_compare.py`.
- Config knobs:
  - `quantize.pack_order in {state_dict,name,size_desc}`

Results:
- `pack_state_dict_struct`:
  - artifact_size_mb: 0.1595
  - val_loss: 2.5642
- `pack_name_struct`:
  - artifact_size_mb: 0.1596
  - val_loss: 2.5642
- `pack_size_desc_struct`:
  - artifact_size_mb: 0.1591
  - val_loss: 2.5642

Decision:
- Keep `size_desc` as the best packing order.
- Do not treat packing order as a standalone frontier mover.

Learnings:
- Sorting by descending tensor size is consistently the best of the three tested pack orders.
- The improvement is small relative to the structured-pruning effect itself.

Next:
- If we keep iterating, combine structured pruning with `size_desc` packing rather than searching packing alone.

## 2026-03-20 - Grouped-query attention architecture sweep

Context:
- Tested the smallest architecture change that could plausibly help without turning the toy model into a new codebase.

Changes:
- Files changed: `toy_model/model.py`, `toy_model/train.py`, `toy_model/run_arch_sweep.py`, `toy_model/config_gqa_best.yaml`, `toy_model/README.md`.
- Config knobs:
  - `model.attn_kv_heads in {1,2,4}`

Results:
- `attn_kv_heads=4` baseline:
  - artifact_size_mb: 0.1604
  - val_loss: 2.6503
- `attn_kv_heads=2`:
  - artifact_size_mb: 0.1583
  - val_loss: 2.6084
- `attn_kv_heads=1`:
  - artifact_size_mb: 0.1556
  - val_loss: 2.6173

Decision:
- Keep `attn_kv_heads=2` as the best architecture candidate.

Learnings:
- Grouped-query attention helped size without the quality penalty you’d expect from a blunt head reduction.
- `kv_heads=1` was too aggressive for this toy setup.

Next:
- If we keep going, combine `attn_kv_heads=2` with the best structured pruning recipe and stop if the gain is marginal.

## 2026-03-20 - Architecture plus compression combo

Context:
- Combined the best architecture knob from the GQA sweep with the best structured pruning recipe.

Changes:
- Files changed: `toy_model/run_arch_combo.py`, `toy_model/config_arch_combo_best.yaml`, `toy_model/README.md`.
- Config knobs:
  - `model.attn_kv_heads=2`
  - `prune.mode=row`
  - `prune.include_patterns=["attn"]`
  - `prune.amount in {0.08,0.10,0.12}`
  - `quantize.pack_order=size_desc`

Results:
- `prune.amount=0.08`:
  - artifact_size_mb: 0.15744781494140625
  - val_loss: 2.6036025881767273
- `prune.amount=0.10`:
  - artifact_size_mb: 0.15835094451904297
  - val_loss: 2.608449548482895
- `prune.amount=0.12`:
  - artifact_size_mb: 0.1588296890258789
  - val_loss: 2.6158049404621124

Decision:
- Keep `prune.amount=0.08` as the current combined arch+compression candidate.

Learnings:
- The GQA gain survives when paired with structured pruning.
- More pruning past 0.08 just erodes quality faster than it saves bytes.

Next:
- Stop broad architecture search here unless the leaderboard context suggests a specific new knob worth testing.

## 2026-03-20 - RunPod 1xH100 real-pipeline reference run

Context:
- Ran `train_gpt.py` on RunPod secure cloud with 1x H100 to get a real baseline for this repo on cached FineWeb (`sp1024`).

Changes:
- Infra/runtime only; no code changes.
- Pod profile: secure cloud, 1x H100, `torchrun --standalone --nproc_per_node=1`.
- Dataset command used: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.
- Training command used:
  - `RUN_ID=... DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=600 torchrun --standalone --nproc_per_node=1 train_gpt.py`

Results:
- stop_step: `1133/20000` (wallclock cap at `600271ms`)
- step_time_avg_ms: `529.81`
- peak_mem: `allocated=10239 MiB`, `reserved=10730 MiB`
- final_exact:
  - `val_loss=2.27510323`
  - `val_bpb=1.34744428`
- artifact sizes:
  - `Total submission size=67,279,676 bytes`
  - `Total submission size int8+zlib=12,881,371 bytes` (within 16 MB limit)

Decision:
- Keep as current real-run reference baseline for 1xH100.

Learnings:
- Current pipeline is stable on RunPod and produces a valid compressed artifact under the challenge limit.
- Quality is substantially better than short smoke runs and is suitable as a baseline for next tuning passes.

Next:
- Compare against an 8xH100 competition run and evaluate scaling/per-dollar tradeoff.

## 2026-03-20 - Cleanup no-go probes and test hidden-state distillation

Context:
- User requested the next paper-style compression idea while removing experiments we no longer need in code.
- Chosen next idea: TinyBERT-style intermediate hidden-state distillation.

Changes:
- Removed no-go probe code paths and scripts:
  - `toy_model/run_wanda_probe.py` (already removed), `toy_model/run_awq_lite_probe.py`, `toy_model/run_mla_probe.py`
  - removed Wanda pruning path from `toy_model/prune.py` and `toy_model/train.py`
  - removed AWQ-lite protection knobs from `toy_model/quantize.py`, `toy_model/size_report.py`, and `toy_model/train.py`
  - removed MLA latent-KV branch from `toy_model/model.py` and `toy_model/train.py`
- Added hidden-state distillation support:
  - `toy_model/model.py` now supports `return_hidden=True`
  - `toy_model/train.py` adds optional `distill.hidden_enabled` + `distill.hidden_weight`
  - `toy_model/run_hidden_distill_probe.py` created
- Removed stale no-go temp/probe artifacts from `toy_model/tmp_*` and `toy_model/runs/*probe*` for Wanda/AWQ/MLA.

Results:
- Hidden-distill probe (run via `.venv_pg`):
  - baseline logits-only (`hidden_probe_base`): `artifact_size_mb=0.1587`, `val_loss=2.5415`
  - hidden weight `0.05` (`hidden_probe_w005`): `artifact_size_mb=0.1543`, `val_loss=2.7142`
  - hidden weight `0.10` (`hidden_probe_w010`): `artifact_size_mb=0.1592`, `val_loss=2.7436`
- Summary file: `toy_model/runs/hidden_distill_probe_summary.json`

Decision:
- Keep code cleanup.
- Drop hidden-state distillation for now (clear quality regression vs logits-only baseline).

Learnings:
- On this toy setup, extra hidden alignment over-constrains the compact student and hurts final loss.
- The existing logits distillation remains stronger for current objective.

Next:
- Keep `toyfocus` baseline and test only new mechanisms with clear size-loss upside.

## 2026-03-20 - ALiBi positional encoding probe (paper idea #1)

Context:
- Implemented requested next idea: replace learned positional embeddings with ALiBi bias.
- Goal: reduce bytes and improve stability/quality at toyfocus settings.

Changes:
- Files changed:
  - `toy_model/model.py` (added `model.positional_encoding` with ALiBi support)
  - `toy_model/train.py` (plumbed config and metrics logging)
  - `toy_model/config.yaml`, `toy_model/config_distill_qat.yaml`, `toy_model/config_toyfocus_best.yaml`
  - `toy_model/run_alibi_probe.py`
  - docs/trackers updated

Results:
- Probe summary: `toy_model/runs/alibi_probe_summary.json`
- Learned positional embeddings:
  - parameter_count: `168,128`
  - artifact_size_mb: `0.159216`
  - val_loss: `2.541510`
- ALiBi:
  - parameter_count: `159,936`
  - artifact_size_mb: `0.156304`
  - val_loss: `2.385367`
- Delta (ALiBi vs learned):
  - `delta_artifact_size_mb = -0.002912`
  - `delta_val_loss = -0.156143`

Decision:
- Keep ALiBi.
- Promote `config_toyfocus_best.yaml` to `model.positional_encoding=alibi`.

Learnings:
- In this toy setup, ALiBi is a strict win on both size and loss.
- Removing learned position embeddings reduced params and helped compact frontier quality.

Next:
- Run a small follow-up around ALiBi with tiny prune/temp perturbations to confirm robustness.

## 2026-03-20 - Mixed-bit quantization probe on ALiBi baseline (paper idea #4)

Context:
- Implemented layer-wise mixed-bit quantization to test if it complements the new ALiBi baseline.

Changes:
- Files changed:
  - `toy_model/quantize.py`: added bit-packing for non-4-bit (`2..7`) and pattern-based per-layer bit overrides
  - `toy_model/size_report.py`: plumbed `quant_layer_bits` through artifact-size estimation
  - `toy_model/train.py`: reads `quantize.layer_bits`, applies in QAT fake-quant and payload estimate, logs metrics
  - `toy_model/run_mixedbit_probe.py`: 7-case mixed-bit probe runner (desktop-safe with 2 workers)
  - Added compact preset: `toy_model/config_toyfocus_mixedbit_compact.yaml`

Results:
- Probe summary: `toy_model/runs/mixedbit_probe_summary.json`
- Baseline (`quant_bits=4`, ALiBi):
  - artifact_size_mb: `0.157160`
  - val_loss: `2.385367`
- Best compact variant (`quant_bits=3`):
  - artifact_size_mb: `0.144482`
  - val_loss: `2.388274`
- Delta (int3 vs int4 baseline):
  - `delta_artifact_size_mb = -0.012678`
  - `delta_val_loss = +0.002907`
- Selective override sets (`quantize.layer_bits`) were neutral in this toy setup.

Decision:
- Keep mixed-bit capability.
- Keep `config_toyfocus_best.yaml` as quality-first baseline.
- Keep `config_toyfocus_mixedbit_compact.yaml` as compact-size preset.

Learnings:
- ALiBi + global int3 is a strong compact point with negligible loss regression.
- Fine-grained per-layer overrides did not beat simple global int3/int4 tradeoff in current toy model.

Next:
- Move to next high-upside mechanism: attention-map distillation (`#2`) or 2:4 sparsity with sparse-aware packing (`#3`).

## 2026-03-20 - Attention-map distillation probe on ALiBi baseline (paper idea #2)

Context:
- Implemented teacher-student attention-map distillation to replace the earlier no-go hidden-state distillation path.

Changes:
- Files changed:
  - `toy_model/model.py`: optional return of per-layer attention probabilities
  - `toy_model/train.py`: added attention distillation (`distill.attn_enabled`, `distill.attn_weight`) with teacher/student head-alignment
  - `toy_model/config.yaml`, `toy_model/config_distill_qat.yaml`, `toy_model/config_toyfocus_best.yaml`, `toy_model/config_toyfocus_mixedbit_compact.yaml`
  - `toy_model/run_attn_distill_probe.py`

Results:
- Probe summary: `toy_model/runs/attn_distill_probe_summary.json`
- Baseline logits-only:
  - artifact_size_mb: `0.157887`
  - val_loss: `2.385367`
- Attention distill `attn_weight=0.02`:
  - artifact_size_mb: `0.157223`
  - val_loss: `2.373386`
  - delta vs base: `-0.000665 MB`, `-0.011981 loss`
- Attention distill `attn_weight=0.05`:
  - artifact_size_mb: `0.158058`
  - val_loss: `2.383788`
  - delta vs base: `+0.000171 MB`, `-0.001578 loss`

Decision:
- Keep attention-map distillation at `attn_weight=0.02`.
- Promote it into `config_toyfocus_best.yaml`.

Learnings:
- Attention-map supervision is materially better than hidden-state MSE for this compact toy model.
- A light attention loss gives the best tradeoff; heavier weight trends toward diminishing returns.

Next:
- Implement and test `#3`: 2:4 structured sparsity with sparse-aware packing.

## 2026-03-21 - 2:4 pruning plus sparse-aware payload packing (paper idea #3)

Context:
- Implemented and evaluated 2:4 structured pruning with custom sparse payload encoding.

Changes:
- Files changed:
  - `toy_model/prune.py`: added deterministic `nm_2_4_prune_model`
  - `toy_model/train.py`: supports `prune.mode=nm2_4` and deferred 2:4 prune after QAT loop
  - `toy_model/quantize.py`: added sparse 2:4 payload packing path (`quantize.sparse_2_4_pack`)
  - `toy_model/size_report.py`: plumbed sparse pack knob
  - `toy_model/run_sparse24_pack_probe.py`
- Config knobs:
  - `prune.mode=nm2_4`
  - `quantize.sparse_2_4_pack=true|false`

Results:
- Probe summary: `toy_model/runs/sparse24_pack_probe_summary.json`
- Baseline (`alibi + attn_distill + int4`):
  - artifact_size_mb: `0.158091`
  - val_loss: `2.373386`
- 2:4 (`attn+mlp`) with sparse packing:
  - artifact_size_mb: `0.130073`
  - val_loss: `3.017763`
- 2:4 (`attn-only`) with sparse packing:
  - artifact_size_mb: `0.149537`
  - val_loss: `2.405295`

Decision:
- Keep sparse 2:4 capability in code.
- Do not promote as default branch (current ALiBi + int3 / attn-distill branches remain better quality-size frontier).

Learnings:
- Sparse packing itself works and gives meaningful byte reduction once a strict 2:4 mask is present.
- Quality degradation from enforced 2:4, especially on broad layer sets, is the limiting factor.

Next:
- Return to lighter knobs on current best stack (teacher/temperature micro-retune).

## 2026-03-21 - Tiny teacher/temperature retune on current best stack

Context:
- Ran a minimal sweep around current ALiBi + attention-distill baseline to capture cheap quality gains.

Changes:
- Added `toy_model/run_teacher_retune_probe.py`
- Tuned only:
  - `distill.teacher_steps in {200,240,280}`
  - `distill.temperature in {2.0,2.2,2.4}`
- Added new preset:
  - `toy_model/config_toyfocus_quality_best.yaml`
- Updated:
  - `toy_model/config_toyfocus_best.yaml` to balanced winner

Results:
- Probe summary: `toy_model/runs/teacher_retune_probe_summary.json`
- Baseline (`240`, `2.2`):
  - artifact_size_mb: `0.158091`
  - val_loss: `2.373386`
- Quality winner (`200`, `2.0`):
  - artifact_size_mb: `0.1592`
  - val_loss: `2.2548`
- Balanced winner (`280`, `2.0`):
  - artifact_size_mb: `0.1523`
  - val_loss: `2.3452`

Decision:
- Promote balanced winner to default `config_toyfocus_best.yaml`.
- Keep quality-max variant as `config_toyfocus_quality_best.yaml`.

Learnings:
- Small distillation schedule tweaks still move the frontier materially.
- The `temp=2.0` branch dominated `2.2/2.4` in this local toy setting.

Next:
- If staying toy-only, run a very tight 4-case confirmation around the new balanced winner.

## 2026-03-21 - Tight confirmation sweep on toyfocus best

Context:
- Ran the recommended 4-case confirmation around the newly promoted balanced preset.

Changes:
- Added `toy_model/run_toyfocus_confirm_probe.py`
- Swept:
  - `teacher_steps in {260,280,300}`
  - `temperature in {1.9,2.0}`
- Updated default:
  - `toy_model/config_toyfocus_best.yaml` now uses `teacher_steps=300`, `temperature=2.0`

Results:
- Summary: `toy_model/runs/toyfocus_confirm_probe_summary.json`
- Prior baseline (`280`, `2.0`):
  - artifact_size_mb: `0.1529`
  - val_loss: `2.3452`
- New best (`300`, `2.0`):
  - artifact_size_mb: `0.1525`
  - val_loss: `2.2995`
  - delta vs baseline: `-0.0004 MB`, `-0.0457 loss`

Decision:
- Keep and promote `teacher_steps=300`, `temperature=2.0` as current default toyfocus best.

Learnings:
- The 2.0 temperature branch remains strongest.
- Slightly stronger teacher (`300`) still improves quality without size penalty.

Next:
- If we continue toy-only, run one small seed-variance check (2 seeds) for this exact config.

## 2026-03-21 - Multi-token prediction (MTP) integration and probe

Context:
- Implemented MTP auxiliary loss to test DeepSeek-style multi-token prediction benefits on toyfocus best.

Changes:
- Files changed:
  - `toy_model/train.py`: added `mtp` config path and auxiliary loss computation
  - `toy_model/config*.yaml`: added `mtp.enabled`, `mtp.weight`, `mtp.horizons`
  - `toy_model/run_mtp_probe.py`
- Promoted config:
  - `toy_model/config_toyfocus_best.yaml` now uses `mtp.enabled=true`, `mtp.horizons=[2,3]`, `mtp.weight=0.1`

Results:
- Summary: `toy_model/runs/mtp_probe_summary.json`
- Baseline (MTP off):
  - artifact_size_mb: `0.1528`
  - val_loss: `2.2995`
- Best MTP (`horizons=[2,3]`, `weight=0.1`):
  - artifact_size_mb: `0.1517`
  - val_loss: `2.2887`
  - delta vs baseline: `-0.0011 MB`, `-0.0108 loss`

Decision:
- Keep MTP in default toyfocus best config.

Learnings:
- Single-horizon MTP (`[2]`) was harmful in this setup.
- Multi-horizon supervision (`[2,3]`) provided a small but real frontier gain.

Next:
- Optional: run 2-seed robustness check on the new default config.

## 2026-03-21 - 4-seed dual-preset robustness decision

Context:
- Needed to decide whether to keep searching new toy algorithms or lock the stack and promote.

Changes:
- Added `toy_model/run_dual_preset_seed_check.py`
- Compared:
  - `quality` preset (`config_toyfocus_best.yaml`)
  - `compact` preset (`config_toyfocus_mixedbit_compact.yaml`)
- Seeds: `1337, 2027, 3141, 4242`

Results:
- Summary: `toy_model/runs/dual_preset_seed_check_summary.json`
- Quality:
  - mean loss: `2.1837`, stdev: `0.0624`
  - mean size: `0.1558 MB`, stdev: `0.0026`
- Compact:
  - mean loss: `2.2318`, stdev: `0.0937`
  - mean size: `0.1463 MB`, stdev: `0.0011`
- Auto-recommendation: `quality`

Decision:
- Keep `quality` preset as default.
- Keep `compact` as optional size-first mode.
- Do not add more toy algorithms immediately; move effort to real-pipeline promotion/A-B validation.

Learnings:
- Quality preset has better average loss and better stability in this 4-seed check.
- Compact preset remains useful when byte pressure dominates.

Next:
- Port the proven toy stack knobs to `train_gpt.py` and run short baseline-vs-promoted A/B.

## 2026-03-21 - Toy scale-up workflow + budget guardrail

Context:
- Closed remaining scale-up checklist items by moving toy runs beyond tiny corpus and enforcing tighter runtime iteration limits.

Changes:
- Files changed: `toy_model/build_local_corpus.py`, `toy_model/config_scaleup_desktop.yaml`, `toy_model/run_scaleup_guardrail.py`, `toy_model/train.py`, `toy_model/README.md`, `toy_model/EXPERIMENT_TRACKER.md`, `checklist.md`.
- Config knobs:
  - New `train.time_budget_sec`, `train.stop_on_budget`, `train.budget_check_every`
  - New scale-up preset with larger model (`d_model=256`, `n_layers=4`, `max_seq_len=128`) and compression path.

Results:
- Built larger local corpus: `toy_model/data/local_slice_corpus.txt` (`~2,000,139` bytes).
- Guardrail run (`steps=50`, `max_seconds=25`, `.venv_pg`):
  - `guard_base`: `artifact_size_mb=1.5613`, `val_loss=3.4248`, `train_time_sec=1.2569`, `time_budget_hit=false`
  - `guard_scaleup`: `artifact_size_mb=0.8036`, `val_loss=3.4956`, `train_time_sec=26.5422`, `time_budget_hit=true`

Decision:
- Keep. Scale-up path and budget simulation are now operational and documented.

Learnings:
- Budget checks must include all training phases (main loop + QAT/teacher), not just the primary loop.
- Larger architecture with compression can still stay far below 16 MB on toy pipeline.

Next:
- Run the same guardrail on CUDA desktop to retune `batch_size`/`time_budget_sec` for GPU throughput.
- If frontier stalls, introduce a new mechanism inspired by recent Qwen/DeepSeek papers (e.g., expert routing, grouped low-rank adapters, or better distillation targets).
