# Parameter Golf Checklist

Progress tracker for local development and challenge readiness.

## Setup

- [x] Create `toy_model/` scaffold.
- [x] Add process docs: `agent.md`, `memory.md`.
- [x] Fix Python/PyTorch environment (`.venv_pg` + torch 2.4.1+cpu).
- [x] Verify `python toy_model/train.py --config toy_model/config.yaml --run-name smoke` runs successfully.

## Baseline Loop

- [x] Tiny dataset file exists (`toy_model/data/tiny_corpus.txt`).
- [x] Baseline model and train loop implemented.
- [x] Metrics output implemented (`parameter_count`, `artifact_size_mb`, `train_time_sec`, `val_loss`).
- [x] Hard 16MB artifact check implemented.
- [x] Record first successful baseline metrics in `memory.md`.

## Compression Features

- [x] Quantization utilities implemented (`toy_model/quantize.py`).
- [x] Magnitude pruning implemented (`toy_model/prune.py`).
- [x] Weight-sharing option implemented (`model.weight_sharing`).
- [x] Low-rank replacement implemented (`toy_model/lowrank.py`).
- [x] Remove no-go compression branches from active code (Wanda/AWQ-lite/MLA).
- [x] Add and test hidden-state distillation probe (`run_hidden_distill_probe.py`).
- [x] Add and test ALiBi positional encoding (`model.positional_encoding=alibi`).
- [x] Add and test mixed-bit quantization (`quantize.layer_bits`, int3/int4 probe).
- [x] Validate each feature with successful runs.
- [x] Record per-feature tradeoffs in `memory.md`.

## Ablations

- [x] Ablation runner exists (`toy_model/run_ablations.py`).
- [x] Run all ablations end-to-end.
- [x] Review `toy_model/runs/ablation_summary.json`.
- [x] Decide keep/drop for each technique.

## Scale-Up

- [ ] Move from toy corpus to larger local dataset slice.
- [ ] Tune config for desktop GPUs.
- [ ] Add timing budget simulation for stricter iteration.
- [ ] Re-check artifact-size behavior under larger configs.

## Challenge Prep

- [ ] Review current official submission requirements in `README.md` of challenge repo.
- [ ] Draft `records/...` submission folder structure.
- [ ] Prepare execution script, logs, and concise write-up.
- [ ] Dry-run packaging and reproducibility checks.
- [ ] Submit PR when metrics and constraints are satisfied.

## Ongoing Discipline

- [x] Append every meaningful experiment result to `memory.md`.
- [x] Keep `checklist.md` status updated after each session.
