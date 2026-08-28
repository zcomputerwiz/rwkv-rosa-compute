# Experiment 0B seed study: result

**Status**: complete. Pre-registered, confirmed, independently reviewed.
**Analysis**: [`scripts/analyze_0b_seed_study.py`](../scripts/analyze_0b_seed_study.py)
**Pre-registration**: [`experiment0_0b_preregistration.md`](experiment0_0b_preregistration.md),
frozen 2026-08-24, before the remaining seeds were evaluated.

Does the RWKV-7 filler-token effect replicate across seeds? Nine runs: five at
`N=0` silent tokens, four at `N=36`.

## Result

Per-seed `corrupted_negative_near_3plus` ROC AUC at the pinned epoch 5.

```text
  arm  seed             run_id      AUC  transition
    0    42   c968fce9af66aa32   0.7095         yes
    0    43   706b5459779b201d   0.6211          no
    0    44   d6d23abcab7a898b   0.7493         yes
    0    45   0c3f9edbcb2c310f   0.7598         yes
    0    46   304c24dc614f6b1a   0.6998         yes
   36    42   c923f49572cadb88   0.5671          no
   36    44   cd865b1f9c9b1089   0.5588          no
   36    45   cf9e58a1052dc20a   0.5474          no
   36    46   e1ee93fa823e4523   0.5585          no
```

**Primary**, one-sided exact Mann-Whitney U, alternative `N=0 > N=36`:

```text
p = 0.0079    REJECT at alpha = 0.05
```

`0.0079` is `1 / C(9,5)`, the **smallest p-value this design can produce**. The
arms separate completely: the lowest `N=0` endpoint (`0.6211`) exceeds the
highest `N=36` endpoint (`0.5671`). No rank assignment could have done better,
and none could have done better by a wider margin.

**Secondary**, descriptive only and not a significance claim: 4 of 5 `N=0` seeds
showed a transition against 0 of 4 at `N=36`, Fisher two-sided `p = 0.0476`.

## What "pre-registered" means here

The analysis script was committed before the remaining seeds were evaluated, so
it could not be tuned to the data. It pins, and refuses to proceed without:

- the outcome **stratum** and the **epoch** (5) — not "the best epoch present",
  which would let a still-training seed contribute a mid-flight value;
- evaluation **settings** (`batch_size=128`, `precision=bf16`), because batch
  size moves the reported AUC by about `0.003`;
- the challenge set by **content hash**, not by identifier — a regenerated set
  could reuse an id while holding different instances;
- the study population by **`run_id`**, not seed number, because earlier result
  families reuse seed numbers at different configurations.

If any endpoint is missing, or if two evaluations of the same checkpoint
disagree, the script withholds the primary result rather than reporting a
partial one.

## Limitations

### Evaluation hardware is confounded with arm

All five `N=0` evaluations ran on an RTX 4060 Ti (`sm_89`) and all four `N=36`
on an RTX 3070 Laptop (`sm_86`), with no checkpoint evaluated on both. This was
disclosed before any outcome was observed and is not removable retrospectively.

The effect is real, not hypothetical. With bit-identical parameters and inputs,
three devices produce three different bf16 logit digests:

```text
device            sm     param_sha256   logits_sha256
RTX 4060 Ti       89     e9f8f20b...    a3d1f26c...
RTX 3070 Laptop   86     e9f8f20b...    1fbd3bfa...
RTX 2070          75     e9f8f20b...    0502d55b...
```

[`scripts/cross_node_numerics_probe.py`](../scripts/cross_node_numerics_probe.py)
reproduces this in seconds per node and requires no data transfer. **The
magnitude on trained checkpoints was not measured**, and no bound on it is
claimed — see the tie density below for why one could not be constructed
cheaply.

**A replicator should avoid creating this confound rather than measure it.**
Evaluate both arms on the same device. Where multiple devices are needed, keep
each seed whole on one machine but assign seeds so every device evaluates both
arms. Never map arm one-to-one onto device, which is what produced the confound
here. Record device identity in every evaluation artifact.

### The outcome metric is extremely tie-dense

Each artifact records 6,000 per-instance rows across six strata. The AUC is
computed over 3,000 of them: 2,000 positives — every row whose realized label is
true, from both positive strata — against the 1,000
`corrupted_negative_near_3plus` negatives. Those 3,000 margins take very few
distinct values:

```text
  arm  seed   distinct   % of rows in a repeated value   adversarial tie bound
    0    42        264                            95.8                0.023101
    0    43         54                            99.4                0.082302
    0    44        266                            95.6                0.023492
    0    45        190                            98.1                0.022855
    0    46        181                            97.4                0.028386
   36    42         85                            99.1                0.040186
   36    44         78                            99.1                0.047161
   36    45         12                           100.0                0.136175
   36    46         68                            99.1                0.053392
```

Tie-corrected ranks are therefore **part of the outcome definition**, not an
implementation detail. The analysis uses averaged ranks on both sides.

The last column is the exact worst-case shift if every tie resolved
adversarially, `0.5 * sum(p*n over tied margins) / (P*N)`. Two of nine endpoints
exceed the smallest cross-arm gap (`0.6211 - 0.5671 = 0.0540`) and a third
essentially equals it. That is why no bound on the hardware term's effect is
claimed: any such bound would have to survive this tie structure, and a cheap
one does not.

This does not weaken the result. Ties are resolved identically for both arms by
a rule fixed in advance; the numbers above describe an adversary who is free to
choose, which is not the situation. They are recorded because a reader
evaluating how much slack the metric has deserves to see them.

## Reproducing it

The analysis is pure standard library — no torch, no numpy — and runs in
seconds on CPU:

```bash
python scripts/analyze_0b_seed_study.py --eval-dir <artifact directory>
```

Evaluation artifacts are **not in this repository**: `results/` is gitignored,
and the per-epoch artifacts are roughly 1.9 MB each. They are distributed
through the project's shared folder under `exp0_0b_seed_study/`, 45 files
covering both arms at epochs 1-5, each with a `.sha256` sidecar.

Pass `--eval-dir` pointing only at that directory. The default also searches a
local `results/`, and anything there stops the run being a reproduction from
published inputs.

## Claims withdrawn during review

Recorded because earlier drafts circulated and should not be quoted:

- a bound of `0.002` on the hardware term's effect on the outcome metric;
- the derived claim that the term was "26x too small to matter";
- the conclusion that hardware "cannot affect the conclusion".

The bound applied a bf16 ULP to the already-formed margin rather than to the two
logits before subtraction. That understates the perturbation by roughly two
orders of magnitude, and it assigns zero possible movement to every row whose
margin is exactly zero — 189 of the 3,000 rows entering seed 42's AUC, where
`true_logit = false_logit = 23.125` and the per-logit ULP is `0.125`, so the
margin could in fact move by up to `0.25`. It also presented a Monte Carlo
maximum as an adversarial bound. The confirmed result never depended on any of
it.

## Review history

The pre-registration, the analysis implementation, and this result were reviewed
by `codex-shannon` and `opencode-dijkstra`. Review found and fixed seven defects
in the analysis script, including tie handling, epoch pinning, and the
conflicting-evaluation gate. The hardware limitation above is the disposition
reached after a proposed stopping argument was rejected and withdrawn.
