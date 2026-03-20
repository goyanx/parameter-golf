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
