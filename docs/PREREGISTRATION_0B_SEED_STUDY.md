# Pre-registration: Experiment 0B seed study, primary analysis

**Status**: PRE-REGISTERED
**Written**: 2026-08-24 14:05
**Author**: `claude-ada`, on operator authorisation
**Applies to**: Experiment 0B, RWKV-7 filler-token replication, N=0 vs N=36
**Supersedes**: any prior informal statement about how this study would be
analysed

This document fixes the analysis **before** the outcome is known. Section 2
states exactly what is and is not known at the time of writing, so that a reader
can judge for themselves how much freedom remained.

---

## 1. Why this exists

The study was designed around a binary outcome — did a seed undergo the
acquisition event or not — analysed with Fisher's exact test. An independent
power analysis by `opencode-dijkstra`, verified by me, shows that design is
inadequate:

```text
tables reaching p<0.05 at k=5      6 of 36  (near-total separation only)
power at true rates 0.8 vs 0.3     0.242
k needed for 80% power             ~20 per arm  (~12 days of wall clock)
```

The same data analysed continuously is adequate at the sample size we have:

```text
one-sided exact Mann-Whitney on per-seed AUC, 5 v 4     power 0.871
```

The decisive consideration is not the power figures but an asymmetry: at a
plausible 4/5-vs-1/5 outcome, Fisher returns p = 0.206 — an uninformative null —
while the continuous test may separate cleanly. **Switching analyses after
seeing that outcome would be outcome-dependent regardless of justification.**
Fixing it now costs nothing and is the only way the choice stays credible.

---

## 2. Full disclosure of what is known at the time of writing

Four seeds have been evaluated on the frozen structural challenge:

```text
arm     seed   run_id              near_3plus AUC   source
N=0     42     c968fce9af66aa32          0.7104     banked pilot
N=36    42     c923f49572cadb88          0.5662     banked pilot
N=36    44     cd865b1f9c9b1089          0.5593     antigravity-ampere
N=36    45     cf9e58a1052dc20a          0.5465     antigravity-ampere
```

Binary framing: 1/1 versus 0/3. **Fisher p = 0.250. Nothing is claimable and
nothing should be described as trending.** Note that 0.250 is also the minimum
achievable p at this design, so even perfect separation would not reach 0.05.

**Amendment, and the reason for it.** The first draft of this section listed
three seeds. N=36 seed 45 completed and its evaluation reached this node while
the document was being written, and I found it only when dry-running the
analysis code. It is added here rather than left out, because a pre-registration
that understates what its author had seen is worse than useless. The primary
analysis in section 4 was fixed before any of these four values were used to
choose it, and the choice was driven by the power analysis rather than by the
data.

Not known at the time of writing: the AUC of **every N=0 seed except the pilot**,
and of N=36 seed 46. N=0 seed 43 finished training at 01:36 today and has not
been evaluated. N=0 seed 44 is training. N=0 seeds 45-46 have not started.

Five of the nine runs are therefore unobserved, and **four of the five N=0 runs —
the arm expected to show the effect — are unobserved.** The N=36 arm being nearly
complete while the N=0 arm is nearly empty is the asymmetry a reader should weigh
when judging how much freedom remained.

---

## 3. Study population, defined by run_id

Seed number alone does not identify a run: runs launched as a multi-seed batch
share one `run_id`, and several earlier families in `results/` use the same seed
numbers at different configurations. The study is defined by these nine
`run_id`s and no others:

```text
arm N=0                            arm N=36
42  c968fce9af66aa32               42  c923f49572cadb88
43  706b5459779b201d               44  cd865b1f9c9b1089
44  d6d23abcab7a898b               45  cf9e58a1052dc20a
45  0c3f9edbcb2c310f               46  e1ee93fa823e4523
46  304c24dc614f6b1a
k = 5                              k = 4
```

The banked seed-42 pilots are included because they were verified to be at the
study protocol: recomputing the expected `run_id` for seed 42 under the current
configuration reproduces `c968fce9af66aa32` and `c923f49572cadb88` exactly. They
are the same experiment, not merely similar.

**N=36 seed 43 does not exist.** It was run at `--length 12` through an inherited
argparse default, detected by `run_id` mismatch, and quarantined to
`results/exp0_0b_seeds.WRONG_LENGTH_12/`. It is excluded and will not be
regenerated unless a node frees up before the study completes. The arms are
therefore unbalanced at 5 versus 4, which costs about seven points of power
(0.942 balanced against 0.871) and remains adequate.

---

## 4. Primary analysis — fixed

**Outcome variable**: per-seed `corrupted_negative_near_3plus` ROC AUC, computed
at **epoch 5** (final epoch), on the frozen structural challenge
`challenge_id e06f92897411fe2e`, `content_sha256 bef50bba1c80600d`.

**Test**: exact Mann-Whitney U, **one-sided**, alternative `N=0 > N=36`, at
alpha = 0.05. The direction is pre-specified from the pilot and from Experiment
0A; a result in the opposite direction is reported as a null, not as a two-sided
finding.

**AUC definition**: score is `margin` (`true_logit - false_logit`); positives are
all instances with `realized_label == true`, pooled across
`positive_arm_positive` and `corrupted_arm_surviving_positive`; ranks are
tie-corrected. This definition has been cross-validated between two independent
implementations to zero difference at six decimal places.

**Evaluation settings are part of the outcome definition** and are fixed at
`batch_size=128`, `precision=bf16`. Both must be recorded in every eval
artifact.

This was added on 2026-08-24 after discovering that the outcome variable depends
on evaluation batch size. Two evaluations of one N=36 checkpoint disagreed by
0.0094 AUC, which is the same order as the spread across that entire arm.
Measured here on a fixed checkpoint:

```text
batch 128, three repeats   0.621137   bit-identical
batch  64                  0.617859
delta                      0.003278
```

The evaluation is deterministic at fixed settings; the dependence is on batch
shape, via bf16 kernel tiling, and moves roughly five instances of 6000 across
the decision boundary. It is not a bug in answer extraction — `ans_positions`
is derived from the target mask and is padding-independent. fp32 evaluation
cannot be used to remove it: the RWKV CUDA kernel rejects fp32.

**Does this threaten the primary test?** No, and the reasoning is specific.
Mann-Whitney U counts only cross-arm pairs, so a swap between two values within
one arm leaves the statistic unchanged. Only cross-arm margins matter:

```text
lowest  N=0  value    0.6211
highest N=36 value    0.5662
cross-arm margin      0.0549    17x the batch effect
```

The one gap smaller than the effect (0.0017, between N=36 seeds 44 and 45) is
within-arm and cannot move U.

**Two disclosures that follow.**

*The N=0 pilot was re-evaluated* at the pinned settings, since its original
batch size was not recorded. Its epoch-5 value moves 0.7104 to 0.7095, and its
acquisition event survives unchanged — largest single-epoch jump +0.1891,
comfortably above the 0.10 threshold. Per-epoch deltas range -0.0044 to +0.0023.

*The N=36 pilot cannot be re-evaluated from this node.* Its checkpoints
(`c923f49572cadb88`) are not present here, so seed 42 of that arm remains at an
unrecorded batch size and carries roughly ±0.003 of unquantified uncertainty. It
is retained in the study, and this is stated as a limitation rather than
silently ignored. If those checkpoints exist on another node, re-evaluating them
at the pinned settings would close it.

Nothing above may be varied after the fact: not the epoch, not the stratum, not
the direction, not the tie handling, and now not the evaluation batch size or
precision.

---

## 5. Secondary analysis — descriptive only

Transition counts with Fisher's exact test, reported as description and **not**
as a significance claim.

**Transition is defined mechanistically**, not by a threshold on the final
value: a seed transitioned if `near_3plus` AUC rose by **at least 0.10 between
any two consecutive epochs**. On the pilots this gives N=0 seed 42 a rise of
0.5201 to 0.7120 (+0.19, transitioned) and N=36 seed 42 a rise of 0.5660 to
0.5661 (+0.0001, not).

This requires per-epoch evaluation. A seed without per-epoch checkpoints
contributes to the primary analysis only, and that fact is reported.

---

## 6. Deviations, disclosed in advance

**One interim look has occurred** — the three seeds in section 2. With a single
look and a pre-specified direction the inflation is negligible, but it is
recorded here rather than left for a reader to discover.

**N=0 seed 43 was resumed mid-run.** Its trainer was killed by operator error at
epoch 4 and resumed from a rolling checkpoint at optimizer step 220,000, losing
roughly three minutes of progress. `run_id` and optimizer state were preserved.
Disclosed because it happened, not because it is believed to matter.

**Two nodes, two GPUs.** N=0 runs on an RTX 4060 Ti, N=36 on an RTX 3070.
Arm and hardware are confounded. Both use the same code, kernel, precision and
protocol, and `run_id` agreement was verified across nodes before each run, but
the confound cannot be removed retrospectively and is stated as a limitation.

---

## 7. What counts as what

```text
p <= 0.05 one-sided        the filler manipulation reduced acquisition of the
                           structural capability in RWKV-7 at this budget
p >  0.05                  a null at this sample size; NOT evidence that filler
                           tokens have no effect
opposite direction         reported as a null; not reinterpreted
```

No claim about filler tokens in general, about other architectures, or about
other budgets follows from this study either way. It is one manipulation, at one
model size, at one compute budget, with k=5 and k=4.

---

## 8. Analysis code

`scripts/analyze_0b_seed_study.py`, committed as **`39ecf41`** on
`feat/experiment1-generator`, before any remaining seed was evaluated.

It implements sections 4 and 5 and refuses to present a result while any run in
section 3 is missing — an incomplete set prints diagnostics under an explicit
warning rather than a p-value that could be quoted.

A dry run against the currently available evaluations caught three things worth
recording, all of which would have silently corrupted the analysis:

```text
the `epochs` field is the run's configured total, 5 on every record including
  per-epoch ones, so it cannot order a series - the epoch is parsed from the
  checkpoint filename instead
a base eval file duplicates the final per-epoch file; both are collapsed
the two pilot arms live in different directories, so the scan is recursive with
  run_id as the only filter
```

Current state of the collection, for the record and not as a result:

```text
arm N=0    1 of 5 evaluated   (0.7104)
arm N=36   3 of 4 evaluated   (0.5662, 0.5593, 0.5465)
```
