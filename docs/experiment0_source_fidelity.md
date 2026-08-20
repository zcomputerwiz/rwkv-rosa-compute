# Experiment 0 Match-3 source-fidelity pass

This document records the source-level fidelity pass performed after the first
repaired length-6 controls. It complements
`experiment0_positive_control_repair.md` and `experiment0_execution.md`.

The goal is not to reproduce every accidental implementation artifact in the
reference repository. The goal is to reproduce the behaviors that materially
define the Match-3 training distribution, supervision stream, and loss geometry,
while keeping deviations explicit and testable.

## Why another fidelity pass was required

The repaired 100k length-6 controls separated several effects:

- the immediate arm could fit its training set but generalized only weakly;
- CoT-only training learned pair-slot identity perfectly;
- unmatched pair-sum generation remained at its structured random baseline;
- matched-index accuracy plateaued near two thirds;
- CoT-only training spent most wall-clock time waiting for Python data work.

Comparing those results against the checked-in `JacobPfau/fillerTokens` Match-3
implementation exposed additional protocol differences.

## Data construction

The default replication generator is now:

```text
generator_mode = source_corrupted
true_rate      = 0.5
corruption_rate = 4/3
```

`true_rate` is the probability of selecting the planted-positive *construction
arm*. It is not the final positive-label rate in `source_corrupted` mode.

The planted arm:

1. samples two random tuples;
2. constructs their modular inverse as a third tuple;
3. appends random remaining tuples;
4. shuffles the complete instance.

The corrupted arm:

1. starts from the same planted three-row core;
2. samples `min(Geometric(1/corruption_rate), 3)` corruptions;
3. applies the source NumPy mixed slice/advanced-index assignment semantics;
4. appends random remaining tuples;
5. shuffles;
6. computes the actual final Match-3 label.

The checked-in probabilistic source path does not successfully reject corrupted
examples that still contain a valid triple. Reproducing that behavior is
important: at larger lengths, the realized majority-class baseline can be
substantially above 50%. Reports therefore continue to compute the baseline from
the realized validation labels.

The previous repository generator remains available explicitly as:

```text
generator_mode = uniform_conditioned
```

That mode guarantees the requested True/False class and is useful for unit tests
or ablations, but it is not the default replication distribution.

### Independent RNG streams

The protocol layer samples the full planted/corrupted construction vector before
creating tuple contents, using a deterministic RNG stream separate from the tuple
generator. This mirrors the source structure and preserves a useful invariant:
requesting a larger dataset does not change the examples in the common prefix.

Generator mode, construction rate, and corruption rate are part of run and
sweep identity. Offline `generate_data.py` artifacts encode the same fields in
their output path and metadata.

## Dense parallel CoT match ordering

For a pair `i < j`, a matching third tuple is searched only in the suffix
`k > j`.

Thus a sorted solution triple `i < j < k` is exposed exactly once:

```text
(i, j) -> k
```

rather than redundantly as all three unordered pairs. This matches the checked-in
dense solver and prevents later teacher-forced match tokens from trivially
recycling an earlier certificate for the same triple.

The filler budget remains `n^2`.

## Fused reduced-CoT tensorization

The reduced parallel-CoT hot path now constructs all of the following in one
pair traversal:

- target token IDs;
- exact diagnostic type;
- semantic-valid target IDs;
- pair index;
- stochastic result-NLL floor.

Previously the dataset formatted and split a Python string, then traversed all
pairs again to recreate diagnostic metadata. The fused path retains an exact
formatter-equivalence regression while removing that duplicate arithmetic.

## Source-compatible output loss geometry

The reference Llama configuration retains a 32,000-class language-model head
even though Match-3 uses only a small subset of label IDs. Custom Match-3 vectors
are supplied through `inputs_embeds`, so the stock embedding table is bypassed,
but the 32k classifier participates in cross-entropy.

Experiment 0 now separates:

```text
vocab_size        = compact task token-ID domain
output_vocab_size = classifier width
```

The runner defaults to:

```text
output_vocab_size = 32000
```

without inventing thousands of fake input features. Direct unit-test models can
leave `output_vocab_size=None` and retain a compact classifier.

For 0B, the same synthetic-task output-head width is used so Llama/RWKV compare
backbones under the same task interface; the stock RWKV language-model embedding
and head are still not loaded as part of the pretrained backbone.

## Separator and EOS supervision

The target stream keeps the separator boundary and now also appends a supervised
EOS target after the final True/False token, matching the reference label
construction.

Token ID zero is used for this EOS target. Padding also carries token ID zero in
the input tensor, but padded labels remain `-100`, so only the real post-answer
EOS contributes to loss.

Both separator and EOS supervision are required protocol invariants and therefore
participate in run identity through `Task3SumConfig`.

## Diagnostics added

Mixed-format online training answer accuracy is now split by realized format:

```text
parallel_cot
filler
serial_cot
immediate
neutral
```

This prevents a mixed run from hiding a chance filler classifier behind a
teacher-forced CoT answer shortcut.

CoT diagnostics now also include:

- structured chance baselines;
- expected stochastic result-token NLL floor;
- per-`(i,j)` pair metrics;
- aggregate pair-position, sum, match-index, and result metrics.

The evaluator accumulates these on-device and transfers the statistics once per
validation pass, preserving the CUDA synchronization contract.

## Deliberate non-replications of apparent code artifacts

Two checked-in implementation oddities are not treated as scientific protocol:

1. `probabilistic_dense_solve()` contains a stray extra final coordination item
   after the formal lexicographic pair list, using a stale previous-pair sum.
   The paper defines the parallel CoT in terms of the pairwise `i < j` list, so
   Experiment 0 retains exactly that list.
2. The reference vectorizer contains unused/colliding feature columns caused by
   implementation indexing details. Experiment 0 preserves the intended shared
   tuple-position/digit <-> CoT-position/digit feature semantics rather than
   duplicating those accidental collisions.

If either artifact is later studied, it should be introduced as an explicit
ablation rather than silently folded into the replication condition.

## Acceptance gates on this branch

The branch adds regressions for:

- independent source-construction oracle agreement;
- corrupted-arm examples that remain truly positive;
- pre-sampled construction-vector ordering;
- dataset-size-stable common prefixes;
- exact `k > j` certificate ordering;
- fused tensorization versus the string formatter;
- supervised EOS;
- task vocabulary versus 32k output-head separation;
- per-format training metrics;
- per-pair diagnostics, chance baselines, and stochastic NLL floor;
- generator/output-head fields changing run identity;
- offline dataset artifact provenance.

These tests still need to run in GitHub CI before the branch is suitable for
merge.
