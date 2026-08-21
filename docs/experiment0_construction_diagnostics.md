# Experiment 0 construction-prior diagnostics

Supplementary diagnostics that partition canonical validation by how each
instance was *constructed*, rather than only by whether the model answered it
correctly. Nothing here changes the canonical metric, the validation
distribution, training data, or run identity.

## The construction arm is not the label

The source-faithful generator chooses a construction arm before it builds an
instance:

```text
positive arm   plant a Match-3 solution, then fill the remaining rows
corrupted arm  plant a solution, then apply 1-3 geometric corruptions to it
```

The corrupted arm is strongly associated with the negative label, but it does
not determine it. Corruption can fail to destroy the planted solution, leaving
an instance that came from the corrupted arm and is nevertheless mathematically
positive. `generate_source_instance` therefore computes and returns the
*realized* label rather than assuming the arm's intent, and the published
distribution's >50% baseline at larger lengths is a direct consequence.

Those survivors are the population this document is about. They are the examples
where the generation cue points one way and the mathematics points the other:

```text
construction prior  →  negative
realized label      →  positive
```

## Why idx1743 motivated this

The completed N=0 run (`filler accuracy 0.993`, 14 errors of 2000) contained one
error of that exact shape:

```text
idx 1743
realized label       True
prediction           False
construction arm     corrupted
corruption count     2
valid triples        1
first witness        [1, 3, 4]
```

Corruption was applied twice and the solution survived anyway. It was the only
error in that run where the construction arm and the realized label disagreed.

**A single such error does not establish shortcut learning, and this diagnostic
does not claim it does.** One instance is not evidence of a systematic prior. The
point of making this a first-class stratum is to ask a question that *can* be
answered: do prior-conflicting examples become systematically easier as the
filler budget grows? That requires the same stratum measured across an N sweep,
with enough examples in it to support a comparison.

## Strata

Every validation instance is assigned one primary stratum:

```text
positive_arm_positive              planted, realized positive
corrupted_arm_negative             corrupted, realized negative
corrupted_arm_surviving_positive   corrupted, realized positive  <-- the point
positive_arm_negative              structurally impossible; present so it
                                   surfaces rather than crashing
unknown_construction_arm           provenance was not recorded
```

Corrupted instances are subdivided by corruption count:

```text
corrupted_positive_c1  corrupted_positive_c2  corrupted_positive_c3
corrupted_negative_c1  corrupted_negative_c2  corrupted_negative_c3
```

`num_valid_triples`, `multiple_witnesses`, and `first_witness` are retained per
instance so they can be aggregated later. This release deliberately does **not**
define a scalar difficulty score.

## Usage

```bash
python scripts/run_experiment.py ... --construction_diagnostics
python scripts/analyze_exp0_errors.py <report>.json --errors --compare
```

For deterministic reconstruction from completed training checkpoints and
cross-seed hard-instance overlap, see
[`experiment0_checkpoint_analysis.md`](experiment0_checkpoint_analysis.md).

`--construction_diagnostics` reuses the run's existing final validation pass, so
it adds no forward pass. It records per-stratum counts, errors, accuracy, and
confusion cells, plus a per-instance record for every error including the True
and False logits and their margin. Empty strata report `accuracy: null` rather
than `0.0`, which would read as total failure rather than absence of data.

## Observed strata from the N=0 10M run

```text
group                                     n   errors   accuracy
overall                                2000       14     0.9930
positive_arm_positive                   948        7     0.9926
corrupted_arm_negative                  952        6     0.9937
  corrupted_negative_c1                 700        4     0.9943
  corrupted_negative_c2                 190        2     0.9895
  corrupted_negative_c3                  62        0     1.0000
corrupted_arm_surviving_positive        100        1     0.9900
  corrupted_positive_c1                  87        0     1.0000
  corrupted_positive_c2                  12        1     0.9167
  corrupted_positive_c3                   1        0     1.0000
```

Read the small strata carefully. `corrupted_positive_c2` shows 8.3% error, which
looks dramatic and is one error out of twelve. Comparing surviving positives
against positive-arm positives gives `1/100` versus `7/948`, Fisher two-sided
**p = 0.55** — no evidence of a difference. That is the expected state of this
diagnostic at a single N with canonical sampling, and it is exactly why the
challenge set exists.

## The diagnostic challenge set

Surviving positives are only 5% of canonical validation, and the c2/c3
subdivisions are far rarer still. A challenge set suppresses the construction
prior by requesting a fixed count per stratum:

```bash
python scripts/run_experiment.py ... --construction_diagnostics --challenge_per_class 1000
```

It uses the unmodified source-faithful generator and keeps every accepted
instance exactly as generated; only *which* instances are retained differs, by
rejection sampling. At length 6 roughly 11% of corrupted draws survive, so the
rare stratum costs about nine draws per accepted instance. Provenance records the
spec, seed, requested and realized strata, attempts and acceptance rate per
stratum, and a deterministic `challenge_id`.

**The challenge set is intentionally distribution-shifted.** Its accuracy is not
comparable to `filler_accuracy` and must never replace or be averaged with it.
Reports keep the two under separate keys for that reason:

```text
construction_diagnostics.canonical_validation
construction_diagnostics.diagnostic_challenge_validation
```

It is evaluation-only. Do not train on it, tune against it, early-stop on it, or
select checkpoints by it. If it starts driving research decisions repeatedly, a
second held-out diagnostic set should be created.

## Interpreting small strata

The strata that matter most are the smallest ones, which is the central
statistical hazard here. `compare_strata` reports counts and error rates
alongside an exact Fisher test, and annotates any comparison whose smaller
stratum has fewer than 30 examples as descriptive only. A large percentage on a
handful of examples is not a finding. Do not describe a subgroup difference as
significant on the strength of its raw rate.

## What would constitute evidence

The eventual curves are not only overall filler accuracy against N, but:

```text
positive_arm_positive             accuracy vs N
corrupted_arm_negative            accuracy vs N
corrupted_arm_surviving_positive  accuracy vs N
corrupted_positive_c2             accuracy vs N
```

Per-run JSON is emitted in a stable shape with every known stratum present even
when empty, so a sweep-level aggregation can align columns across N without
parsing console output. Aggregation itself is deliberately left to a later
change.

