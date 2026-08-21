# Experiment 0 checkpointing and resume

Long Experiment 0 runs can write exact-resume training checkpoints without changing the scientific run identity. Checkpoint paths and save cadence are operational settings and are intentionally excluded from `run_id`.

## Default policy

`scripts/run_experiment.py` defaults to:

```text
--checkpoint_every_steps 5000
```

When checkpointing is enabled, each seed writes under:

```text
<out_dir>/checkpoints/<run_id>/seed_<seed>/
```

The trainer maintains:

```text
latest.pt       rolling recovery point, overwritten atomically
epoch_001.pt    permanent snapshot after epoch 1 validation
epoch_002.pt    permanent snapshot after epoch 2 validation
...
```

Periodic `latest.pt` checkpoints are written only after a completed optimizer step and are skipped on the final batch of an epoch because the post-validation epoch checkpoint immediately follows. Epoch checkpoints are written after filler validation and CoT diagnostics have completed.

Set `--checkpoint_every_steps 0` to disable automatic checkpoint creation. Passing an explicit `--checkpoint_dir` still enables epoch checkpoints, and `--resume_checkpoint` can resume even when the new periodic cadence is zero.

## What is saved

A training checkpoint contains enough state to continue at the next unprocessed sample without repeating an optimizer update:

- model parameters;
- AdamW optimizer state;
- LR scheduler state;
- AMP GradScaler state when FP16 is active;
- completed-epoch history and best metrics;
- partial current-epoch loss/answer/format accumulators;
- current epoch, optimizer step and consumed-sample offset;
- the explicit shuffled-epoch seed;
- Python, Torch CPU and Torch CUDA RNG states;
- initialization provenance;
- a versioned compatibility signature for the requested model/training/task configuration.

The training dataset itself is regenerated from the normal Experiment 0 seed. Dataset formatting is deterministic by `(dataset seed, sample index)`, so storing prefetched batches is unnecessary. The resumable sampler regenerates the same epoch permutation and starts at the saved sample offset; indices prefetched before a crash but not yet optimized are therefore replayed rather than skipped.

Checkpoint files are loaded with `weights_only=False` because they intentionally contain optimizer, scheduler, RNG and metric state. Resume only checkpoints produced by a trusted Experiment 0 run.

## Resume

Resume exactly one training seed by supplying the same experiment arguments plus a checkpoint path:

```powershell
python scripts/run_experiment.py `
  <same scientific/training arguments> `
  --seeds 42 `
  --resume_checkpoint results\exp0\checkpoints\<run_id>\seed_42\latest.pt
```

`--resume_checkpoint` requires exactly one `--seeds` value. The trainer refuses a checkpoint if the saved compatibility signature differs from the requested model, optimizer/training protocol, task configuration, dataset size, realized format assignment, epoch count, or run ID. The checkpoint cadence itself may be changed on resume because it does not affect the optimization trajectory.

## Continuing Completed Runs (`continue_training.py`)

A run that has already completed its planned epoch budget cannot simply be resumed with `--resume_checkpoint`. The `epochs` count is part of the checkpoint compatibility signature (so asking for more is rejected), and `linear_warmup_decay` has reached `0.0` (so training at the stored learning rate would perform no parameter updates).

Extending a completed run requires a **new learning rate schedule**, which is an intervention. To answer exploratory questions ("does the metric move further if training continues?"), use `scripts/continue_training.py`:

```powershell
python scripts/continue_training.py results/exp0/checkpoints/<run_id>/seed_42/epoch_005.pt `
  --additional-epochs 2 `
  --device cuda `
  --out results/continuations/seed_42_extended.json
```

### Safety and Provenance Rules for Continuations

1. **Reconstructed Configuration**: All model, dataset, task, and optimizer settings are reconstructed directly from the checkpoint's internal `signature` rather than re-specified via CLI flags. A continuation cannot accidentally train on a different data distribution or model than the run it extends.
2. **Restored Optimizer State**: Both model weights and AdamW optimizer momentum (`exp_avg`, `exp_avg_sq`) are restored to device memory; only the learning rate schedule is new. The peak LR defaults to the source run's last nonzero learning rate.
3. **Supervised Loss Target**: Cross entropy is computed strictly against `loss_mask` (with `-100` ignore index) rather than `targets` (which contains padding tokens), ensuring the model is not penalized on padded positions.
4. **Non-Canonical Output**: The resulting report is explicitly marked:
   ```json
   "is_canonical_experiment_result": false
   ```
   Continuations are exploratory artifacts and **must not** be placed on an accuracy-vs-N curve alongside fixed-budget runs.

## Atomicity and recovery behavior

`latest.pt` is written to a temporary file in the same directory, flushed, fsynced and atomically replaced. A process or machine failure during a new save therefore leaves either the previous valid `latest.pt` or the newly completed one rather than a deliberately half-written target file.

Epoch snapshots are saved once and then atomically copied to `latest.pt`. The permanent epoch files are retained so a damaged or accidentally overwritten rolling checkpoint does not remove all recovery points.

## Testing

`tests/test_exp0_checkpointing.py` and `tests/test_continue_training.py` include tests for:

- atomic replacement;
- RNG capture/restore;
- exact sampler suffix replay;
- incompatible-checkpoint rejection;
- simulated mid-epoch process loss followed by exact resume;
- continuation schedule construction, optimizer state restoration, and non-canonical metadata enforcement.

## DataLoader settings are not part of run identity

Worker count, prefetch depth, and memory pinning change how batches are
produced, never what they contain. `Task3SumDataset.__getitem__` derives every
item from `(seed, idx)` alone — the format code is precomputed per index and the
per-item RNG is `random.Random(f"{seed}_{idx}")` — so an index yields the same
example whichever worker builds it, and `DataLoader` preserves batch order for a
fixed sampler.

They were nonetheless fingerprinted into the `run_id`, which meant giving a run
more workers made it, by convention, a different experiment. That is now fixed:

```text
num_workers, val_num_workers, pin_memory, prefetch_factor
  -> normalized to their defaults in drop_identity_neutral_fields
```

Normalized rather than removed, so the canonical config keeps the same shape.

### Resuming across a change

`validate_checkpoint_signature` accepts a checkpoint whose only disagreements
are these fields, emits a `RuntimeWarning`, and proceeds.

`run_id` is exempt too, but **only as a consequence** and only when every other
section already matches. The `run_id` is a hash of exactly the model, task,
evaluation, and training inputs in the signature; if all of those agree and the
only training disagreements are DataLoader fields, the differing hash can only
have come from those fields. The exemption is a deduction, not an override — a
`run_id` mismatch alongside any substantive difference is still rejected, and so
is a mixture of loader drift and a real protocol change.

### Runs whose identity shifted

Three completed runs used `num_workers=2` and therefore hash differently now:

```text
74e021c82a17c377  ->  b8d95a7a...
9e81911f97b19e87  ->  44eb57e0...
ef1125605d565142  ->  951e3d23...
```

Their checkpoints remain resumable — that is what the tolerance above is for —
but recomputing the identity of those configs no longer reproduces the recorded
directory or filename. Runs with the default `num_workers=0` are unaffected;
verified against every result file in `results/`.

The practical consequence worth knowing: a run and its more-parallel rerun now
share a `run_id`, and therefore an output filename. That is the intended
semantics — they are the same experiment — but the later run overwrites the
earlier report rather than sitting beside it.
