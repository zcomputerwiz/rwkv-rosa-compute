# Experiment 0B seed study: result

**Status**: **closed.** Statistically successful and scientifically useful;
mechanistically inconclusive. No further 0B training is planned.

Pre-registered and independently reviewed. The reported statistic is confirmed
for the collected artifacts. Evaluation-device numerics have since been measured
and are too small to explain the separation; **training-path and seed-by-device
attribution remains unresolved** — see Limitations.

**Analysis**: [`scripts/analyze_0b_seed_study.py`](../scripts/analyze_0b_seed_study.py)
**Pre-registration**: [`PREREGISTRATION_0B_SEED_STUDY.md`](PREREGISTRATION_0B_SEED_STUDY.md),
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
it could not be tuned to the remaining five endpoints. It could not be blind to
the first four: section 2 of the pre-registration lists them, with their AUCs,
as what was already known at the time of writing. It pins, and refuses to
proceed without:

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

### Hardware is confounded with arm

All five `N=0` runs were trained and evaluated on an RTX 4060 Ti (`sm_89`) and
all four `N=36` on an RTX 3070 Laptop (`sm_86`), with no registered checkpoint
crossing devices. Arm is confounded with hardware in both training and
evaluation.

Post-registered work has since separated those two halves. The evaluation half
is measured and small. The training half is not removable retrospectively and
remains the operative limitation.

The confound was disclosed in the pre-registration, which was written after four
of the nine endpoints had been evaluated — section 2 of that document lists them
— and before the remaining five were run.

The effect is real, not hypothetical. With bit-identical parameters and inputs,
three devices produce three different bf16 logit digests:

```text
device            sm     param_sha256   logits_sha256
RTX 4060 Ti       89     e9f8f20b...    a3d1f26c...
RTX 3070 Laptop   86     e9f8f20b...    1fbd3bfa...
RTX 2070          75     e9f8f20b...    0502d55b...
```

[`scripts/cross_node_numerics_probe.py`](../scripts/cross_node_numerics_probe.py)
reproduces this in seconds per node and requires no data transfer.

**The magnitude on a trained checkpoint has since been measured.** One fixed
`N=0` seed-44 epoch-5 checkpoint, evaluated on all three devices under the
pinned protocol:

```text
RTX 4060 Ti  sm_89    0.74928125    reference
RTX 3070     sm_86    0.75039850    +0.00111725
RTX 2070     sm_75    0.74879975    -0.00048150
                      range          0.00159875
```

That is an observed range for this checkpoint, challenge and protocol — not a
universal numerical bound. Per-instance outputs are **not** invariant: 105 of
6,000 structural predictions flip between two of the devices. The rank-based
endpoint absorbs that; individual predictions do not. Evaluation-device
numerics therefore cannot account for the registered separation, which is a
narrower and stronger statement than the earlier "not measured".

### Training hardware remains confounded, and is the live limitation

Evaluation is settled; training is not. The registered study mapped training
device one-to-one onto arm, and no amount of post-hoc evaluation fixes that.

Two post-registered within-device comparisons exist, both on **seed 44 only**:

```text
device                N=0 AUC     N=36 AUC    arm difference
RTX 3070  sm_86      0.7315830   0.5587728      +0.1728103
RTX 4060 Ti sm_89    0.7492813   0.4620020      +0.2872793
```

Both were produced after the registration and are **one-seed diagnostics**.
They reject GPU identity as the sole explanation for seed 44's separation, and
the sm_89 pair shows the effect emerging during training rather than existing at
the first checkpoint. They do not retroactively randomize training device across
the nine registered runs, and they are not a substitute for a paired-seed
extension.

Three qualifications belong with those numbers.

**The sm_89 pair is same-device but not same hardware state.** The registered
`N=0` run trained at this machine's original memory clock; the `N=36` run
trained at a clock reduced by 300 MHz. It is a same-device, different-clock
comparison.

**A first `N=36` attempt at the original clock diverged.** It reached epoch 2
with 114,981,888 of 115,031,808 parameters non-finite and was discarded. The
completed run is a second attempt at the reduced clock. A transient fault or an
overclock-related cause is **plausible, not proven**: the training path does not
enable deterministic algorithms, and two runs of one seed on this machine were
measured to differ — 400 of 403 tensors, largest delta `5.98e-03` — so a
recipe-level instability would not necessarily have reproduced either.

**Same-arm training outcomes differ materially across nodes.** `N=36` seed 44
scores `0.4620` here and `0.5588` on the RTX 3070: the same arm, seed and
recipe, `0.0968` apart. Since run-to-run variation on one machine has not been
separated from device variation, that gap bounds neither term individually. It
is the reason a paired design — both arms of a seed on one machine — is worth
more than an unpaired seed expansion.

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

A fixed averaged-rank rule removes analyst discretion: nobody chose how these
ties fell. It does not make the metric robust. Because arm and device are
perfectly confounded, device-specific numerics can move rows into and out of
ties, and a tie that breaks differently changes cross-arm ordering. The
adversarial column is not a claim that an adversary exists; it bounds how much
room the metric leaves, and for two endpoints that room exceeds the separation
being reported.

## Reproducing it

The analysis is pure standard library — no torch, no numpy — and runs in
seconds on CPU:

```bash
python scripts/analyze_0b_seed_study.py --eval-dir <artifact directory>
```

Evaluation artifacts are **not in this repository and not publicly
distributed**: `results/` is gitignored and the per-epoch artifacts are roughly
1.9 MB each. They exist in the project's private replicated working folder
(45 files, both arms, epochs 1-5, each with a `.sha256` sidecar), which is
**outside the repository and unavailable to an ordinary clone**. A reader
cloning this repository cannot currently reproduce the number from published
inputs. Publishing a filename plus SHA-256 manifest is the intended next step.

Pass `--eval-dir` pointing only at that directory. The default also searches a
local `results/`, and anything there stops the run being a reproduction from
published inputs.

## WITHDRAWN — DO NOT CITE

These claims circulated in earlier drafts and are **retracted**. A canonical
retraction is safer than leaving circulated claims without one:

- a bound of `0.002` on the hardware term's effect on the outcome metric;
- the derived claim that the term was "26x too small to matter";
- the conclusion that hardware "cannot affect the conclusion".

The bound applied a bf16 ULP to the already-formed margin rather than to the two
logits before subtraction. That understates the perturbation by roughly two
orders of magnitude, and it assigns **zero** possible movement to every row
whose margin is exactly zero — 189 of the 3,000 rows entering seed 42's AUC.
A zero margin does not imply zero perturbation: those rows carry 66 distinct
equal-logit values spanning `15.5` to `25.875`. At the upper end one bf16 step
is `0.125`, so a single logit moving one step shifts the margin by that much,
and independent one-step changes to the two logits in opposite directions can
move it by as much as `0.25`. It also presented a Monte Carlo maximum as an
adversarial bound. The reported statistic never depended on any of it.

## Review history

The pre-registration, the analysis implementation, and this result were reviewed
by `codex-shannon` and `opencode-dijkstra`. Review found and fixed seven defects
in the analysis script, including tie handling, epoch pinning, and the
conflicting-evaluation gate. The hardware limitation above is the disposition
reached after a proposed stopping argument was rejected and withdrawn.


## Closure

Experiment 0B is closed. The registered statistic stands, the evaluation-device
confound is measured and small, and the training-device confound is disclosed
and not removable from the collected population.

Further paired 0B seeds would sharpen a causal claim about this specific filler
recipe at high cost, and would not establish the latent-workspace or
post-query-computation mechanism the programme is actually asking about.
Expected information per GPU hour is now higher in Experiment 1.

**Reopening requires one of:** an external requirement for a population-level
causal claim about the 0B filler arm; an Experiment 1 failure that a narrowly
defined 0B experiment can diagnose; or explicit operator authorization of the
pre-registered paired-seed extension. Absent one of those, new 0B training is
out of scope, and no existing 0B task document should be read as authorizing it.
