# Experiment 1: gate 0a, and the recall-capacity audit

Gate 0a asks the simplest question on the Experiment 1 ladder: at depth `D=1`,
given a start node and one selector, can the model emit that node's image under
that map? It ran, and it failed. This document records what it returned, the
tool built to diagnose it, and — as importantly — the claims that tool does
**not** support.

## 1. What gate 0a returned

Registered regime: `d_model 128`, `layers 2`, batch 64, 10 epochs, fixed;
`V = 16` nodes, `K_maps = 4`, fp32 reference kernel, no workspace; one fixed
train bank of 5,000 memories and one fixed held-out bank of 500, shared across
the three model seeds. Outcome is the **median over seeds of final-epoch
held-out accuracy**, threshold `0.95`, chance `1/V = 0.0625`.

```text
seed   final held-out
1001          0.0625
1002          0.0585
1003          0.0625
median        0.0625
```

Chance, on all three seeds. Training loss moved 2.800 to 2.653 against
`ln 16 = 2.7726`.

## 2. Two failures, and they are not the same failure

The number above is produced by **under-training**, not by memorisation. Gate
0a has no `--lr` flag and inherits `TrainConfig.learning_rate = 1e-4` from
Experiment 0, where it was tuned for a 768-wide model. At that rate the loss
barely leaves `ln 16`.

Raise the learning rate into the band the published literature uses for this
task family — 2e-4 and above — and optimisation works. What then happens is
different and is the subject of this audit: the model **memorises the training
bank instead of learning in-context retrieval**, reaching high or perfect
training accuracy while held-out accuracy stays at chance.

Both are real, and only the second is what the recall audit measures. No report
may use the second to explain the first.

## 3. The tool

[`scripts/audit_recall_capacity.py`](../scripts/audit_recall_capacity.py) sweeps
learning rate, `d_model`, layers, bank size, and the task's `M x K` shape, and
reports held-out accuracy alongside training accuracy on a probe of the training
bank. Held-out accuracy alone cannot separate "has not learned" from "has
memorised"; the pair can.

It is an analysis tool. It is not a gate, it decides nothing, and its artifacts
are estimates.

Three properties worth knowing before reading its output:

- **Both `held_out_final` and `held_out_best` are recorded, and `gap` and `z`
  derive from the final epoch**, because that is the statistic the gates use.
  Deriving them from the maximum builds a selection over 32–128 epochs into a
  number that then reads as an outcome.
- **`z` is clustered at the memory level.** `queries_per_memory` queries share
  one memory, so treating instances as independent understates the standard
  error by `sqrt(queries_per_memory)`.
- **The artifact carries its environment and repo commit.** Cells here vary a
  great deal between runs, and a comparison that cannot see the compute
  capability or the torch and CUDA versions cannot separate architecture from
  environment — or, as section 5b shows, from the model seed.

Audit regime, distinct from the gate's in every one of these fields: batch 256,
`lr 1e-3`, AdamW `betas=(0.9, 0.95)`, weight decay 0.01, gradient clipping 1.0,
`CosineAnnealingLR(T_max=epochs)` stepped per epoch, fp32 reference kernel.

## 4. Bank size, at matched optimiser steps

Data volume and optimiser steps are confounded if epochs are held fixed, since a
larger bank then also gets more steps. Run as a 2×2 at matched steps, on
`sm_89`, `d_model 128`, `layers 2`, final-epoch held-out:

```text
                        4,992 memories            19,968 memories
2,496 steps      held 0.0647  train 0.6189   held 0.0631  train 0.0787
                 loss 1.0598                 loss 2.7651
9,984 steps      held 0.0776  train 1.0000   held 1.0000  train 1.0000
                 loss 3.4e-08                loss 2.2e-08
```

Chance is 0.0625. The bottom-left cell is the point: at 4,992 memories with
matched steps the model reaches perfect training accuracy and a loss of
`3.4e-08` while held-out sits at 0.0776, a gap of 0.92. The top-right cell shows
data alone is not sufficient either — 2,496 steps has simply not trained yet.
Both are required.

Leakage was checked before believing the 1.0000, since exact 1.0 with a
vanishing loss is the shape a leak takes: zero memories and zero queries are
shared between the banks, the most common held-out answer occurs at 0.0798
against a uniform 0.0625, and the memory space at `M=16, K=4` is `16!^4 ≈
1.9e53`. The 4,992 bank is a strict prefix of the 19,968 bank — same generator,
same seed — so these are two points on one curve differing only in size.

A third point at 79,872 memories also reaches 1.0000.

## 5. The bank a cell needs scales with its association count

Swept on `sm_75`, 32 epochs, `lr 1e-3`, final-epoch held-out:

```text
  M   K  assoc   at 4,992   at 19,968
  4   1      4     1.0000      1.0000
  4   2      8     1.0000      1.0000
  4   4     16     1.0000      1.0000
  8   1      8     1.0000      1.0000
  8   2     16     1.0000      1.0000
  8   4     32     0.2433      1.0000
 16   1     16     1.0000      1.0000
 16   2     32     0.1406      1.0000
 16   4     64     0.0603      0.9459
```

At or below 16 associations both bank sizes generalise. At 32 the small bank
fails and the large one is perfect. At 64 — the gate's own cell — the large bank
is still short of 1.0.

Note that `M` is not a clean axis: changing it also changes chance, the class
count, the input width, and so the parameter count. `params` is recorded per
cell for that reason.

Every cell above, and every cell in section 4, is **model seed 1001**. Read
section 5b before treating any of them as a property of the bank size.

## 5b. The cell is bimodal across model seeds

Three model seeds at the `19,968` cell, `d=128, L=2`, 32 epochs, everything
else identical, on three architectures — final-epoch held-out, and the gate
outcome the median would produce:

```text
node       seed 1001   seed 1002   seed 1003    median   0a thresholds applied
sm_89         1.0000      0.1021      0.0681    0.1021   would be FAIL  (< 0.30)
sm_86         1.0000      1.0000      0.0725    1.0000   would be PASS  (>= 0.95)
sm_75         0.9459      0.0837      0.9989    0.9459   would be RETRY (0.30-0.95)
```

**The right-hand column is a counterfactual, not a gate result.** These runs
used 19,968 memories, batch 256, 32 epochs, `lr 1e-3` and compilation;
registered gate 0a uses 5,000 memories, batch 64, 10 epochs and `lr 1e-4`,
uncompiled. Applying 0a's thresholds to audit runs shows how little separates
the three outcome branches at this cell. It is not a report of gate 0a having
been run three times.

Five of the nine runs reach near-ceiling with `train 1.0000` and loss between
5e-08 and 2e-03. The other four sit near chance with `train 0.34-0.54` and loss
between 1.4 and 1.9. **Nothing lands in between.** The failing runs are stuck
partway through fitting the training bank, not generalising poorly from a
fitted one — an optimisation-stability signature rather than a capacity one.

The supportable reading is narrow:

> At this audit configuration the outcome is sharply bimodal and sensitive to
> execution context. Three seeds are too few for the median to be a stable
> statistic.

What the crossed audit does establish is negative: **there is no stable device
ordering, and no seed-only explanation.** Seed 1001 converges on all three
nodes, seed 1002 only on `sm_86`, seed 1003 only on `sm_75` — and a two-node
reading of seed 1002 looked like cross-device agreement until the third node
landed opposite both. The earlier single-seed comparison, which had this cell
reaching the threshold on `sm_89` and not on `sm_75`, is refuted.

Rerun nondeterminism has since been measured and does not explain the table.
Twenty-four repeats at this cell, all at the same commit and configuration:

```text
node                comparisons   reproduced outcome branch   bitwise identical
sm_89 (chipset)          12                12                       8
sm_86                     6                 6                       6
sm_75                     6                 6                       6
```

**Every comparison reproduced its outcome branch.** The four exceptions to
bitwise identity are all on `sm_89`, spread across two seeds, at roughly 0.33
events per run; Fisher against the other nodes gives `p = 0.093`, and against
`sm_89`'s own other seeds `p = 0.067`. So the effect is suggestive of that node
and is not attributable to a particular seed. `sm_89` is also the only
chipset-attached node, so architecture and PCIe topology stay confounded.

One of those four moved a reported value — `0.0681` to `0.0608`, 0.64 clustered
SE, both at chance. It reached a final number rather than only the internal
trajectory, so a result sitting within noise of a threshold should not be
called from a single run, and bitwise artifact matching is not a valid
verification method on that node.

What the audit still cannot establish is a positive claim about node-by-seed
interaction, because there is one *configuration* per node-by-seed cell.
Calling the outcome a coin flip, or saying it tracks neither node nor seed,
goes past the data.

**`19,968` memories is necessary on the evidence, not sufficient.** At 4,992 no
seed on any node has produced generalisation. At 19,968 five of nine do, and
the success probability is unresolved.

That last point is the one with consequences, because the outcome here is
Bernoulli rather than continuous, and a median is the wrong instrument for a
rate. Treating a fresh seed as a draw at the observed `p = 0.556`:

```text
N = 1    P(median converges) = 0.556
N = 3    P(median converges) = 0.583
N = 5    P(median converges) = 0.603
N = 11   P(median converges) = 0.647
```

A median over three seeds buys 2.7 points over a single run, and at `p = 0.5`
exactly no `N` helps at all. Reporting a convergence *rate* with a binomial
interval is the instrument the literature uses for bimodal training outcomes
-- see the mixture treatment in
[arXiv:2502.17356](https://arxiv.org/abs/2502.17356), the ten-seed grok-rate
protocol in [arXiv:2607.05104](https://arxiv.org/html/2607.05104), and the
argument against point estimates under few runs in
[arXiv:2108.13264](https://arxiv.org/abs/2108.13264). The 2×2 in section 4 remains a valid
within-seed comparison — same seed, nested banks, only the size changed — but
"quadrupling the bank takes held-out from chance to 1.0000" describes one draw.

**On the schedule.** Holding the rate constant at this cell and the same 9,984
optimiser steps reads `0.9994` on `sm_75` where cosine reads `0.9459`, with the
subsequent 96 epochs worth only a further `0.0006`. That is a demonstrated
accelerator **for seed 1001**, which converges under cosine on all three nodes
anyway. It has not been run on any of the four failing trajectories, so it is a
plausible mechanism for the table rather than a demonstrated rescue, and the
budget-versus-schedule question is settled only for a seed that was never
failing.

## 5c. Repetition, not distinct-memory count

Two controls at the same 9,984 optimiser steps and the same `d=128, L=2` cell
locate the axis more precisely than bank size does.

**A constant learning rate does move held-out off chance.** Holding `lr 1e-3`
for 256 epochs at 4,992 memories, against a clustered SE of 0.01144:

```text
epochs   1– 32   mean held-out  0.0657    chance 0.0625
epochs  97–128   mean held-out  0.0834    +1.83 SE
epochs 225–256   mean held-out  0.0871    +2.15 SE
```

The departure saturates: doubling the budget from 9,984 to 19,968 steps moved
the plateau by about a third of an SE. Training accuracy is 0.9961 and the loss
0.0265 at the end, so gradients persist and the plateau is not an optimiser
artifact. Against a 0.95 threshold this is a long way from passing, but
"chance" was the wrong description.

`sm_86` ran the identical configuration and tracks it closely:

```text
                    sm_89     sm_86
epochs   1- 32     0.0657    0.0658
epochs  97-128     0.0834    0.0745
epochs 225-256     0.0871    0.0847
```

Both rise monotonically across thirds and settle near 0.085. Note what that
agreement does and does not show: the two runs share model seed 1001 and both
data seeds, so they are near-replicates differing only in hardware. It
establishes that the number is stable, not that it survives a change of seed.

**Removing memorisation removes all learning.** Drawing a fresh bank every
epoch — 4,992 memories per epoch, 128 epochs, 638,976 distinct memories, no
memory seen twice:

```text
held-out final   0.0664     chance 0.0625,  z = +0.34
train accuracy   0.0642     also chance
final loss       2.77261    ln 16 = 2.77259
```

The loss reached the entropy of the uniform distribution within a few epochs
and stayed flat to five decimals for the remaining 120.

```text
bank                     exposures/memory   held-out    train    final loss
4,992 fixed                           128     0.0776   1.0000      3.4e-08
19,968 fixed                           32     1.0000   1.0000      2.2e-08
4,992 fresh per epoch                   1     0.0664   0.0642      2.77261
```

Unlimited distinct data at one exposure each is the worst of the three, below
the small fixed bank. So distinct-memory count is not the axis — **repetition
is necessary**, and between 1 and 32 exposures per memory the model goes from
learning nothing to generalising perfectly. Where in that range is not measured.

The same signature appears in Experiment 0: the streaming arm there took 3.3×
the optimiser steps memorisation needed and never left `ln 13`, while the
fixed-set arm memorised. Same architecture, different generator.

## 6. What this does not establish

**It does not show the gate cannot pass at any budget.** `CosineAnnealingLR`
with `T_max=epochs` anneals the learning rate to approximately zero by the final
epoch, and held-out accuracy in the 128-epoch cell flatlines at 0.077567 over
its last eight epochs — a frozen model, not a converged verdict. Section 5
measures what a constant rate does instead.

**It does not show the gate is device-dependent — and the crossed audit says
it is not.** A single seed per node had the cell reaching 0.95 on `sm_89` and
not on `sm_75`, which is where that claim came from. Three seeds per node
invert the medians (section 5b). Seed variance swamps any between-node
difference, and neither ordering was real.

**It is not the memorisation instrument.** Gate 0c is: 512 fixed `D=4`
instances, train and evaluation identical, outcome on training-set accuracy.
Its stated licence is that a later held-out failure is then not explained by an
inability to optimise or to store. Ad hoc "can it fit a small set" controls run
here duplicate a registered gate and should not be reported as findings about
the apparatus.

**It does not import Zoology's bound.** Zoology's claim is about model
*dimension* scaling with MQAR input sequence length. This tool measures a
*data* bound. What is observed here runs mildly against a pure width reading —
`d` is held at 128 while associations go from 4 to 64, and the failure is
rescued by more memories rather than more width. Treat the parallel as an open
question.

## 7. Reproducing

```bash
python scripts/audit_recall_capacity.py --lrs 1e-3 --memories 4992 19968 --epochs 32 --out results/recall_capacity/sweep.json --label my-node
```

Memory counts must be multiples of `batch_size / gcd(batch_size, queries_per_memory)`
— 64 at the default batch 256 and 4 queries per memory. Under
`--compile --compile-backend cudagraphs` a ragged bank is refused up front
rather than raised from inside a graph replay partway through the first
evaluation.

`--fresh-memories-per-epoch` draws a new training bank every epoch, so no memory
is seen twice and memorising is impossible by construction. Generation costs
about 46 µs per instance, so the effective bank is unbounded for roughly 1% more
wall clock. Note section 5c before reaching for it as a fix: at these budgets it
also removes learning, so a run under this flag that sits at chance is not
evidence about retrieval.

`--lr-schedule constant` holds the rate rather than annealing it. Use it for any
claim about budget: under the default cosine, `T_max=epochs` means a longer run
mostly buys frozen epochs, so "more steps did not help" does not follow from one.
