# Experiment Tracker

## Tried

- Baseline tiny transformer (`config.yaml`): stable reference.
- Aggressive compression (`config_compressed.yaml`): strong size win, quality drop.
- Distillation + QAT (`config_distill_qat.yaml`): quality recovery at higher size.
- Tight-size sweeps (`run_target_sweep.py`): reached ~0.12-0.15 MB, loss around ~3.02.
- Parallel light sweep (`run_parallel_sweep.py --mode light --max-workers 2`):
  - Best quality around ~2.455 loss at ~0.238 MB.
- Parallel mixed sweep (pre-stability-controls):
  - ~0.123-0.142 MB, best loss ~2.989.
- Chassis sweep (`run_parallel_sweep.py --mode chassis --max-workers 2`):
  - Tested `quant_group_size` in `{32,48,64}`
  - Tested `qat.delay_fake_quant_steps` in `{0,20}`
  - Best quality in this sweep: loss ~2.9486 at ~0.1438 MB.
- Frontier sweep (`run_parallel_sweep.py --mode frontier --max-workers 2`):
  - Added distillation temperature and tighter frontier settings
  - Best quality: ~2.493 at ~0.1615 MB
  - Best compact under 2.8: ~2.692 at ~0.1565 MB
- Micro sweep (`run_parallel_sweep.py --mode micro --max-workers 2`):
  - Focused around compact frontier with temp `{2.2,2.3,2.4}` and prune `{0.11,0.12}`
  - Best in sweep: `val_loss~2.644` at `~0.1586 MB`
  - New locked config: `config_micro_best.yaml`
- Refine sweep (`run_parallel_sweep.py --mode refine --max-workers 2`):
  - Added `group_size=40` midpoint and `teacher_steps=240` branch
  - Best in sweep: `val_loss~2.576` at `~0.1587 MB`
  - Best compact under 2.8: `val_loss~2.576` at `~0.1587 MB`
  - `config_micro_best.yaml` updated to this winner
- Tight sweep (`run_parallel_sweep.py --mode tight --max-workers 2`):
  - Tightened to `group_size=40`, `prune in {0.10,0.11}`, `teacher_steps in {240,260}`
  - Best result stayed at `val_loss~2.576` / `~0.1587 MB`
  - `teacher_steps=260` did not beat the `240`-step winner
- Main-script port:
  - Added `COMPRESSION_PRESET=toy_micro_best` / `toy_tight` to `train_gpt.py`
  - Portable settings now flip on progressive pruning and int8 QAT without changing defaults
- Three-path toy research sweep (`run_research_paths.py`):
  - Structured sparsity: row pruning on attention blocks won the compact tradeoff
  - Selective precision: protecting extra layers improved loss but expanded size a lot
  - Selective low-rank: attention-only low-rank improved loss dramatically but was not size-efficient
  - New locked structured preset: `config_structured_best.yaml`
- Structured tightening sweep (`run_structured_tight.py`):
  - Prune amounts `{0.06,0.08,0.10,0.12,0.14}` on attention rows
  - Best loss: `~2.5642` at `~0.1592 MB`
  - Best size: `~0.1591 MB` at `~2.5748` loss
  - No meaningful frontier gain over `config_structured_best.yaml`
- Precision-ranked sweep (`run_precision_ranked.py`):
  - Sensitivity-ranked module keeps identified `pos_emb` and `shared_block.attn.qkv` as the best loss movers
  - Keeping them in higher precision improved loss slightly, but artifact size jumped to `~0.169-0.206 MB`
  - No precision-allocation candidate beat the compact frontier
- Breadth sweep (`run_breadth_sweep.py`):
  - Structured row pruning on attention blocks remained the best frontier-preserving branch in the new sweep
  - `pack_order=size_desc` shaved a small number of bytes off the baseline artifact with unchanged loss
  - Selective low-rank improved loss a lot, but the artifact-size cost stayed too high
- Pack compare (`run_pack_compare.py`):
  - Rechecked the best structured recipe with `pack_order in {state_dict,name,size_desc}`
  - `size_desc` was still the smallest packing order, but the gain was small and did not dethrone the existing structured best
- Architecture sweep (`run_arch_sweep.py`):
  - Added grouped-query attention via `model.attn_kv_heads`
  - `attn_kv_heads=2` was the best tradeoff of the three tested settings
  - `attn_kv_heads=1` was smaller but gave back some quality
  - `attn_kv_heads=4` was the baseline and did not outperform the GQA variants
- Architecture combo sweep (`run_arch_combo.py`):
  - Combined `attn_kv_heads=2` with attention-row pruning and `size_desc` packing
  - Best run: `prune.amount=0.08`, `artifact_size_mb~0.1574`, `val_loss~2.6036`
  - This is the best architecture+compression candidate so far, but it still does not beat the structured-pruning best on loss

## In Progress

- Structured row pruning has plateaued; stop extending that branch.
- Next useful work:
  - compare `train_gpt.py` baseline vs `COMPRESSION_PRESET=toy_micro_best` on a short smoke run
  - decide whether a different compression axis is worth a new toy sweep
  - treat layer-wise precision allocation as plateaued unless a new tensor granularity appears
  - consider combining the best structured-pruning recipe with `quantize.pack_order=size_desc`
  - if we keep going, test structured-pruning plus `size_desc` packing against the current best structured config
  - if architecture continues, keep `attn_kv_heads=2` and tighten only around that setting
  - if combining architecture with compression, start from `config_arch_combo_best.yaml`

## Yet To Try

- Run mixed sweep with new stability controls (`--mode mixed --max-workers 2`).
- Compare `config_stability.yaml` against best prior mixed run.
- Completed: mixed sweep with stability controls (`--mode mixed --max-workers 2`).
- Completed: `config_stability.yaml` baseline stability run.
- Distillation temperature sweep (`2.0`, `2.5`, `3.0`) under fixed 0.145-0.160 MB target.
- Teacher strength micro-sweep while keeping student fixed near best `par_02`.
- Micro-sweep prune target (`0.10`, `0.12`, `0.14`) with progressive schedule.
- Micro-sweep low-rank rank (`56`, `64`, `72`) to shave size with minimal loss drift.
- Try `group_size=32` with slightly reduced prune and stronger teacher to push `<2.9` near ~0.14 MB.
- Test delayed fake quant warmup windows `{10,30}` beyond `{0,20}`.
- Micro-sweep around compact frontier:
  - `temp in {2.2, 2.3, 2.4}`
  - `prune in {0.11, 0.12}`
  - fixed `rank=56`, `group_size=48`, `delay=20`
- Completed: `teacher_steps=260` probe in `run_parallel_sweep.py --mode tight`.
- Completed: `group_size=40` vs `48` tie-breaker in `run_parallel_sweep.py --mode refine`.
- Completed: structured row-prune path sweep in `run_research_paths.py`.
- Completed: selective precision path sweep in `run_research_paths.py`.
- Completed: selective low-rank path sweep in `run_research_paths.py`.
- Completed: structured tightening sweep in `run_structured_tight.py`.
- Completed: layer-wise precision ranking sweep in `run_precision_ranked.py`.
- Completed: breadth sweep in `run_breadth_sweep.py`.
- Completed: structured pack-order compare in `run_pack_compare.py`.
- Completed: grouped-query architecture sweep in `run_arch_sweep.py`.
- Completed: architecture-compression combo sweep in `run_arch_combo.py`.

## Objective Check

- Current best near-size:
  - `~0.1587 MB`, loss `~2.576`
- Structured best:
  - `~0.1592 MB`, loss `~2.564`
- Structured tight best size:
  - `~0.1591 MB`, loss `~2.575`
- Current best with `<2.8` achieved:
  - quality-oriented: `val_loss~2.493` at `~0.1615 MB`
  - compact-oriented: `val_loss~2.692` at `~0.1565 MB`
- Gap to close: structured row pruning did not materially improve size, so the next frontier should come from a different compression axis.
- Layer-wise precision allocation also failed to improve the frontier: small loss gains required materially more bytes.
- Pack-order tuning is a minor byte-level knob, not a major frontier mover by itself.
- The best remaining toy-model direction is still structured pruning first, packing order second.
- The best architecture knob so far is grouped-query attention with `attn_kv_heads=2`.
- The best combined architecture/compression point is `attn_kv_heads=2` plus attention-row pruning at `0.08`.
