# Experiment 0 Positive-Control Repair Plan

This document records the apparatus repair prompted by the August 2026 length-6 and length-12 Llama diagnostic runs.

The observed pattern was:

```text
parallel-CoT final answer accuracy: 1.0
filler final answer accuracy:       chance
immediate final answer accuracy:    chance
```

That pattern is not a clean negative result for filler-token computation. It exposed several mismatches between this repository's Experiment 0 implementation and the positive-control protocol in Pfau, Merrill & Bowman, *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models* (arXiv:2404.15758).

The repair is intentionally limited to Experiment 0 apparatus fidelity and diagnostics. It does not change the H2/H3/H4 research sequence.

## Source-derived requirements

The paper and the authors' public `JacobPfau/fillerTokens` implementation establish the following relevant protocol details:

1. The positive control uses a scaled-down randomly initialized Llama transformer with 4 layers, hidden size 384 and 6 attention heads.
2. Tuple inputs contain hard-coded multi-hot digit and position features.
3. The transformer also receives normal Llama positional information.
4. A boundary token (`P` in the authors' code) separates the multi-hot input vectors from the supervised continuation.
5. The authors wrap the Hugging Face Llama in one shared `nn.Linear` input adapter. Reduced CoT tuple-index and digit features reuse feature coordinates also used by the original tuple inputs.
6. The Llama configuration uses `initializer_range = 0.02`; the custom input adapter is constructed afterward and retains `nn.Linear`'s default initialization.
7. Filler runs use `n^2` repeated filler tokens. This budget is retained exactly.
8. Parallel CoT vocabulary reduction randomly chooses one summand-position token and one coordinate digit, so exact token accuracy has an intentional stochastic ceiling.
9. The authors' evaluator separates intermediate CoT positional-token accuracy, matched-index accuracy and sum-token accuracy. Final answer accuracy on a supplied ground-truth CoT is not by itself a measure of independent 3SUM computation.
10. Their Match-3 training runner uses AdamW with betas `(0.9, 0.95)` and a 5% linear warmup followed by linear decay toward zero.
11. Full published training uses 10,000,000 training samples, 2,000 validation samples, five epochs for filler/CoT and 25 epochs for immediate-answer runs.

Primary references:

- https://arxiv.org/abs/2404.15758
- https://github.com/JacobPfau/fillerTokens

## Phase 1 — Restore Llama positional information

Implement standard Hugging Face Llama split-half rotary position encoding (RoPE) on attention Q/K projections.

Requirements:

- apply positions across the complete tuple-plus-continuation sequence;
- preserve causal SDPA;
- keep RoPE base (`theta`) in `ModelConfig` so it participates in run identity;
- reject invalid odd attention head dimensions;
- add an independent split-half rotation regression and a position-sensitivity regression.

The previous positionless transformer should no longer be considered a valid 0A positive-control implementation.

## Phase 2 — Restore the continuation boundary

Keep the `:` separator as the first target-stream token rather than dropping it during dataset tensorization.

This gives the model the same conceptual boundary role as the paper's `P` token:

```text
multi-hot tuple inputs
        ↓
separator token
        ↓ predicts
first filler / CoT / ANS token
```

Consequences:

- the first intermediate token now participates in next-token supervision;
- all CoT diagnostic slots have an actual preceding target-stream state;
- `ANS` evaluation remains unchanged semantically;
- the protocol is recorded as `Task3SumConfig.include_separator_token = true` and the old separator-dropping mode is rejected.

## Phase 3 — Restore shared Match-3 input features

The pre-repair wrapper used:

```text
tuple multi-hot -> tuple_proj
continuation id -> unrelated target_embed
```

This breaks an important transfer path in the authors' implementation. Their vector dataset feeds both problem vectors and continuation vectors through one input `nn.Linear`, and reduced CoT indices/digits reuse the corresponding tuple-position/digit feature columns.

The repaired seam uses one shared `input_proj`:

```text
tuple digit / position feature ─┐
                               ├─ same input_proj column -> hidden state
corresponding CoT digit / label ┘
```

Special tokens without a tuple-feature analogue receive dedicated input-feature columns.

The efficient implementation does not materialize large target one-hot tensors. A continuation token maps to a feature-column index and obtains the corresponding column of `input_proj.weight`, plus the shared linear bias. This is algebraically identical to applying the linear layer to a one-hot feature vector.

The repaired shared-feature protocol is required and participates in model provenance/run identity.

### Initialization fidelity

For Llama 0A:

```text
Llama backbone Linear weights  ~ Normal(0, 0.02)
output classifier head          ~ Normal(0, 0.02)
shared Match-3 input_proj       PyTorch nn.Linear default initialization
RMSNorm weights                 ones
```

This mirrors the authors' construction order: Hugging Face initializes the Llama model using `initializer_range=0.02`, then their `InputEmbedCausalTransformer` creates the custom input linear separately.

RWKV backbone checkpoint loading remains unchanged; Experiment 0's shared synthetic-task input interface stays randomly initialized.

## Phase 4 — Match the positive-control optimizer schedule

Use the authors' Match-3 optimizer defaults:

```text
optimizer       AdamW
beta1           0.9
beta2           0.95
base LR         1e-4 unless explicitly overridden
warmup          first 5% of optimizer steps
schedule        linear warmup, then linear decay toward zero
weight decay    existing protocol value
```

The implementation keeps the schedule defined for tiny CI runs by using at least one warmup step.

These settings are explicit fields of `TrainConfig`, enter deterministic run identity, and are reported with each seed's history. Precision remains an independent recorded knob; the repository does not silently force FP16 simply because the source runner used mixed precision.

## Phase 5 — Replace the misleading CoT diagnostic

The old metric named `cot_accuracy` measured:

> final True/False accuracy at `ANS` while the ground-truth CoT prefix was already supplied.

In the parallel CoT format, matched pairs emit a tuple-index/letter token while unmatched pairs emit a sum digit. Consequently the supplied CoT contains a direct answer certificate. A model can therefore achieve perfect final answer accuracy without independently learning the pairwise computation.

New reports retain this quantity only as:

```text
cot_answer_given_cot_accuracy
```

and explicitly document that it is a leakage-aware diagnostic, not independent-computation evidence.

The diagnostics fall into two groups, and only the second is evidence of
computation.

Structural/leakage-aware (measure layout, counting and certificate reading):

```text
cot_answer_given_cot_accuracy
cot_pair_position_token_accuracy
cot_pair_position_semantic_accuracy
```

`cot_pair_position_semantic_accuracy` accepts either summand label, so emitting
`labels[i]` at every pair slot scores 1.0 without performing a single pairwise
computation. It measures whether the model tracks which pair index it is on,
which is a layout property.

Computational (measure the pairwise work itself):

```text
cot_sum_token_accuracy
cot_sum_semantic_accuracy
cot_match_index_accuracy
cot_result_semantic_accuracy
cot_result_nll
```

### Baselines for `cot_match_index_accuracy`

Two baselines are reported and they are not interchangeable:

```text
match_index_accuracy                      unconditional; the comparison to use
match_index_accuracy_given_match_known    reference only; not a real baseline
```

The conditional form assumes the guesser already knows the pair matches and
only has to choose `k` among the eligible `k > j` suffix. No measured model is
in that position: it must first decide whether a match exists at all. Match
slots are also a small minority of result slots, so a model that has not
learned 3SUM emits a sum digit at every result slot and scores exactly `0.0`
on match index. Exactly zero is the expected null value here, not a defect.

### Ceiling on `cot_pair_position_semantic_accuracy` in mixed-format runs

Every format shares the tuple prefix and the `:` separator, then diverges at
the first continuation token:

```text
parallel CoT   one of the pair's two labels   parallel_ratio / 2 per token
filler         .                              filler_ratio
neutral        #                              neutral_ratio
immediate      ANS                            immediate_ratio
serial CoT     DIM                            serial_ratio
```

If any single non-CoT format outweighs `parallel_ratio / 2`, the argmax at that
slot is never a CoT label and the first pair scores exactly `0.0` no matter what
the model has learned. Every later pair slot is format-disambiguated by the
token before it. The metric is therefore capped at:

```text
(pair_count - 1) / pair_count
```

Under the default 50/50 CoT/filler mixture at length 6 this is 14/15 = 0.93333,
which both the Llama and RWKV 100k runs reproduce to five digits. A value at
this ceiling means saturated, not "13 of 15 pairs learned". Reports now emit
`cot_pair_position_semantic_ceiling` and `cot_first_slot_format_ambiguous`
alongside the metric.

### Exact versus semantic metrics

With vocabulary reduction enabled, the target generator randomly chooses:

- one of the two pair-position labels; and
- one coordinate of a multidimensional pair sum.

Exact-token metrics intentionally mirror the sampled training target and therefore have stochastic ceilings.

Semantic metrics accept any target that is computationally equivalent for the same pair. They answer the more important diagnostic question:

> Did the model compute a valid pairwise result, independent of which equivalent reduced-vocabulary token happened to be sampled?

`cot_result_nll` measures teacher-forced negative log-likelihood over result slots only. It excludes pair-position tokens and the final answer.

## Phase 6 — Add training-fit visibility

Record answer accuracy from each training batch's existing forward pass:

```text
epoch_online_train_answer_accuracies
best_online_train_answer_accuracy
```

This adds no second model forward.

The word **online** is intentional. A batch is scored before its own optimizer update, so this is a low-overhead fit diagnostic rather than a frozen end-of-epoch pass over the entire training set.

It still usefully distinguishes:

```text
online training answer remains near chance
    → optimization / wiring / capacity concern

online training answer becomes high while validation remains chance
    → generalization / sample-complexity concern
```

In mixed-format runs read `best_online_train_answer_accuracy_by_format` rather
than the pooled number. The parallel-CoT arm contains an answer certificate and
saturates early, which drags the pooled figure well above the filler arm.

### Validation answer histogram

Validation is filler-format only. `epoch_filler_accuracies` and
`best_filler_accuracy` are the only names for it; the former `epoch_val_accuracies`
and `best_val_accuracy` aliases have been removed, because two report keys
holding one number read as corroboration when they agree.

Reports now include the predicted True/False/other histogram at the validation
`ANS` position, plus a `degenerate_predictor` flag:

```text
filler_answer_prediction_counts_per_seed
filler_answer_is_degenerate_any_seed
```

Accuracy alone cannot distinguish a model scoring at `majority_class_baseline`
from one emitting a single constant answer. Check the histogram before treating
any accuracy at or near the baseline as a measurement.

## Phase 7 — Add explicit apparatus gates

### Gate A — structural/unit tests

CPU CI must prove:

- RoPE matches an independent Hugging Face Llama split-half rotation calculation;
- identical content at different positions receives different rotary transforms;
- the default filler budget remains exactly `n^2`;
- the separator is retained and supervised;
- tuple positions/digits and corresponding CoT labels/digits reuse shared input-projection columns;
- Llama backbone/head initialization follows the configured `0.02` range while the input adapter retains `nn.Linear` initialization;
- the warmup/decay lambda matches the positive-control schedule;
- every parallel-CoT diagnostic target belongs to its semantic-valid set;
- a scripted model can score 100% on final answer-given-CoT while scoring 0% on result semantics, proving the diagnostic distinguishes leakage from computation;
- the repaired protocol fields participate in deterministic run identity.

### Gate B — tiny fixed-set overfit

A very small length-3/dimension-1 immediate dataset is trained and evaluated on itself using a small CPU Llama.

Expected result:

```text
best online training answer accuracy >= 95%
validation-on-same-set accuracy         >= 95%
```

Failure means the apparatus cannot even memorize a tiny fixed problem set and larger negative results are uninterpretable.

### Gate C — easy generalization diagnostic

Before another large length-6/length-12 run, run an immediate-answer control on the simplest nontrivial family:

```powershell
python scripts/run_experiment.py `
  --architecture llama `
  --length 3 `
  --dimension 1 `
  --num_filler 0 `
  --format_type immediate `
  --num_samples 10000 `
  --val_samples 2000 `
  --epochs 5 `
  --seeds 42 `
  --batch_size 128 `
  --num_workers 2 `
  --out_dir results/easy_generalization
```

This is a runtime diagnostic rather than a fixed CI threshold because held-out generalization depends on optimization scale and should remain visible as experimental evidence rather than becoming a fragile unit test.

## Phase 8 — Re-run the replication progressively

Do not immediately jump to the published 10M-example run.

After Gates A-C are satisfactory, scale length-6/dimension-3 training geometrically:

```text
100k
300k
1M
3M
10M
```

At each scale inspect together:

```text
online training answer accuracy
filler validation accuracy
CoT pair/result generation diagnostics
CoT result NLL
majority baseline
```

Only after the apparatus produces a credible positive-control learning curve should Experiment 0A be considered validated and Experiment 0B/H2 interpretation resume.

## Post-repair observation at 100k (August 2026)

The repaired apparatus was run at the first rung of the Phase 8 ladder, length 6
/ dimension 3, 100,000 training samples, default 50/50 CoT/filler mixture:

```text
                              Llama       RWKV-7
validation (filler) accuracy  0.524       0.524
majority_class_baseline       0.524       0.524
train answer, parallel_cot    1.0         0.99998
train answer, filler          0.5505      0.5422
cot_answer_given_cot          1.0         1.0
cot_pair_position_semantic    0.93333     0.93333
cot_match_index               0.0         0.0
cot_sum_semantic              0.3088      0.3416   (chance 0.2711)
```

**The repair did not change the observed pattern at this scale, and that is the
expected outcome.** 100k is 1% of the published 10M-sample budget. The two
architectures agree to three or more digits across every metric despite
differing in precision, batch size and epoch count, so these runs carry no
architecture information and no filler-token information.

The two constants that do not move with data are explained above and are not
defects: `cot_pair_position_semantic` is at its mixture ceiling and
`cot_match_index` is at the expected null for a model that has not learned the
task.

Nothing here revises the Phase 1-7 repairs. It records that the repairs are
necessary but not sufficient, and that the apparatus has not yet been shown to
produce a positive control at any scale.

## Mixture caveat

The 50/50 single-model CoT/filler mixture is a choice made by this repository.
The published protocol trains a separate model per format. Mixing has two
consequences that belong with any result reported from a mixed run:

1. The parallel-CoT arm carries an answer certificate, saturates to 1.0 early,
   and dominates the pooled training signal.
2. The shared prefix makes the first continuation token format-ambiguous, which
   is what pins the first pair's position metric at zero.

Single-format runs remove both. Prefer them when the diagnostics are the point
of the run, and mark any mixture-specific result as such.

## Interpretation rule

The pre-repair August 2026 Llama runs are diagnostic artifacts, not negative experimental results. They were executed with a positionless transformer, without the supervised continuation-boundary state, with separate tuple/token input representations that removed the source implementation's shared-feature transfer path, and under different optimizer dynamics. The old CoT metric could also saturate from the supplied CoT answer certificate.

The repaired protocol intentionally preserves the paper's `n^2` filler budget and dense parallel-CoT supervision while making the intermediate computation measurable.

The repaired 100k runs are likewise not negative results. A flat filler curve at
1% of the published sample budget is uninformative in both directions, and no
0A/0B/H1 conclusion should be drawn until the Phase 8 ladder produces a
positive control that separates from `majority_class_baseline`.