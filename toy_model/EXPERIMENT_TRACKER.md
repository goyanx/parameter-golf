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

## In Progress

- Narrowing around 0.145-0.160 MB with stability controls to maintain `<2.8` loss.

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

## Objective Check

- Current best near-size (~0.135 MB): loss ~2.95.
- Current best with `<2.8` achieved:
  - quality-oriented: `val_loss~2.493` at `~0.1615 MB`
  - compact-oriented: `val_loss~2.692` at `~0.1565 MB`
- Gap to close: push compact frontier below `~0.156 MB` while keeping `<2.8`.
