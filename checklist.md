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
- [x] Add and test attention-map distillation (`distill.attn_enabled`, `run_attn_distill_probe.py`).
- [x] Add and test 2:4 pruning + sparse-aware packing (`prune.mode=nm2_4`, `quantize.sparse_2_4_pack`).
- [x] Run tiny teacher/temperature retune probe (`run_teacher_retune_probe.py`).
- [x] Run tight confirmation sweep around current toyfocus best (`run_toyfocus_confirm_probe.py`).
- [x] Add and test multi-token prediction auxiliary loss (`mtp.*`, `run_mtp_probe.py`).
- [x] Run 4-seed dual preset decision check (`run_dual_preset_seed_check.py`).
- [x] Validate each feature with successful runs.
- [x] Record per-feature tradeoffs in `memory.md`.

## Ablations

- [x] Ablation runner exists (`toy_model/run_ablations.py`).
- [x] Run all ablations end-to-end.
- [x] Review `toy_model/runs/ablation_summary.json`.
- [x] Decide keep/drop for each technique.

## Scale-Up

- [x] Move from toy corpus to larger local dataset slice.
- [x] Tune config for desktop GPUs.
- [x] Add timing budget simulation for stricter iteration.
- [x] Re-check artifact-size behavior under larger configs.

## Challenge Prep

- [x] Review current official submission requirements in `README.md` of challenge repo.
- [ ] Draft `records/...` submission folder structure.
- [ ] Prepare execution script, logs, and concise write-up.
- [ ] Dry-run packaging and reproducibility checks.
- [ ] Submit PR when metrics and constraints are satisfied.
- [ ] Configure persistent RunPod storage for 8xH100 runs so logs/artifacts survive pod loss.
- [ ] Authenticate GitHub CLI (`gh auth login`) on this workstation for PR automation.
- [ ] Prepare compute-credit application packet (evidence + links + goals + account details).

## Ongoing Discipline

- [x] Append every meaningful experiment result to `memory.md`.
- [x] Keep `checklist.md` status updated after each session.

