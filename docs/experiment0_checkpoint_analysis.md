# Experiment 0 checkpoint re-evaluation and cross-seed analysis

Completed Experiment 0 training checkpoints can be evaluated without restoring
an optimizer, scheduler, scaler, RNG state, or training loop. This is intended
for post-hoc recovery and reproducibility analysis; it does not alter training,
checkpoint selection, validation distributions, or run identity.

Only evaluate checkpoints produced by a trusted Experiment 0 run. Training
checkpoints contain Python and optimizer state and therefore use PyTorch's
trusted-checkpoint deserialization path. All checkpoint tensors are first loaded
on CPU. Only model parameters are subsequently copied to an explicitly requested
CUDA device.

## Evaluate a checkpoint

When a completed run report is available, use it as the source of evaluation and
challenge provenance:

```powershell
python scripts/evaluate_exp0_checkpoint.py `
  --checkpoint results/n0_seed_repro/checkpoints/<run-id>/seed_44/epoch_005.pt `
  --run_report results/n0_seed_repro/<seed-44-report>.json `
  --device cuda `
  --construction_diagnostics `
  --out <external-output-dir>/seed44-diagnostics.json
```

When the report was lost, required values must be explicit. They are never
silently guessed from defaults:

```powershell
python scripts/evaluate_exp0_checkpoint.py `
  --checkpoint results/n0_seed_repro/checkpoints/<run-id>/seed_43/epoch_005.pt `
  --device cuda `
  --eval_seed 9999 `
  --val_samples 2000 `
  --construction_diagnostics `
  --challenge_per_class 2000 `
  --challenge_seed 20260820 `
  --out <external-output-dir>/seed43-diagnostics.json
```

On CUDA, evaluation defaults to the checkpoint's training precision. CPU
evaluation defaults explicitly to FP32 and records that fallback in provenance.
Use the same device, precision, and batch size for every artifact that will be
compared.

The output is a standalone, versioned JSON artifact containing:

- checkpoint hash, training seed, epoch, and optimizer-step provenance;
- model, task, training, and evaluation configuration;
- a `canonical_validation_id` hashing the generation configuration and exact
  tuple, label, planted-witness, construction-arm, and corruption contents;
- the existing generator-defined `challenge_id`, plus a challenge-content hash;
- one ordered record per canonical and challenge example, including logits,
  signed True-minus-False margin, correctness, construction metadata, and tuple
  contents;
- deterministic structural features for every example, regardless of whether
  the model classified it correctly.

The structural features enumerate every candidate `i < j < k` and retain the
number satisfying zero, one, two, or three coordinates, the count of 2-of-3 near
misses, the maximum matched-coordinate count among non-solutions, and the first
valid witness's zero-based position in lexicographic candidate order. These are
mathematical instance properties, not learned difficulty scores.

## Compare training seeds

```powershell
python scripts/compare_exp0_errors.py `
  <external-output-dir>/seed43-diagnostics.json `
  <external-output-dir>/seed44-diagnostics.json `
  <external-output-dir>/seed45-diagnostics.json `
  --reference-errors "365,491,550,602,730,812,875,1016,1209,1359,1498,1675,1743,1933" `
  --out <external-output-dir>/n0-cross-seed-comparison.json
```

The comparator rejects mismatched canonical IDs, task configurations, model
configurations, training protocols, evaluation execution settings, challenge
IDs, or challenge contents. It reports, for arbitrary numbers of seeds:

- each error set and error rate;
- pairwise intersection, union, Jaccard similarity, independent-error expected
  overlap, observed/expected enrichment, and exact hypergeometric upper tail;
- miss-frequency histograms and every example missed by at least two seeds;
- generic reference-error recurrence with its own chance expectation and exact
  tail probability;
- the same overlap calculations within construction and corruption strata;
- per-seed prediction margins for persistent errors;
- full structural-feature histograms for examples correct in all seeds, missed
  by one seed, and missed by at least two seeds, both overall and within
  corrupted-arm negatives.

Hypergeometric probabilities are descriptive evidence. They do not automatically
declare a mechanism or convert this post-hoc diagnostic into a formal stopping or
selection rule.

Diagnostic artifacts and checkpoints are generated results. Keep them outside
committed source trees unless a separate archival decision explicitly says
otherwise.

