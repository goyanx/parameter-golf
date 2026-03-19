# Parameter Golf Toy Playbook

## Iteration Order

1. Confirm baseline run is stable and reproducible.
2. Enable quantization and check artifact-size drop.
3. Enable pruning and confirm loss impact is tolerable.
4. Enable weight sharing and compare parameter efficiency.
5. Enable low-rank layers and tune rank.
6. Keep only changes that improve size/performance tradeoff.

## Required Metrics Per Run

- `parameter_count`
- `artifact_size_mb`
- `train_time_sec`
- `val_loss`
- `global_sparsity`

## Promotion Criteria

- Do not promote a technique unless it lowers artifact size and keeps loss regression acceptable.
- Keep scripts reproducible and avoid manual steps.

