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
3. The transformer itself also receives normal positional information; the paper explicitly notes additional positional information in the input/CoT stream.
4. A boundary token (`P` in the authors' code) separates the multi-hot input vectors from the supervised continuation.
5. Filler runs use `n^2` repeated filler tokens. This budget is retained exactly.
6. Parallel CoT vocabulary reduction randomly chooses one summand-position token and one coordinate digit, so exact token accuracy has an intentional stochastic ceiling.
7. The authors' evaluator separates intermediate CoT positional-token accuracy, matched-index accuracy and sum-token accuracy. Final answer accuracy on a supplied ground-truth CoT is not by itself a measure of independent 3SUM computation.
8. Full published training uses 10,000,000 training samples, 2,000 validation samples, five epochs for filler/CoT and 25 epochs for immediate-answer runs.

Primary references:

- https://arxiv.org/abs/2404.15758
- https://github.com/JacobPfau/fillerTokens

## Phase 1 — Restore Llama positional information

Implement standard Llama-style rotary position encoding (RoPE) on attention Q/K projections.

Requirements:

- apply positions across the complete tuple-plus-continuation sequence;
- preserve causal SDPA;
- keep RoPE base (`theta`) in `ModelConfig` so it participates in run identity;
- reject invalid odd attention head dimensions;
- add an independent pair-rotation regression and a position-sensitivity regression.

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
- the protocol choice is recorded as `Task3SumConfig.include_separator_token` and therefore changes deterministic run identity.

## Phase 3 — Replace the misleading CoT diagnostic

The old metric named `cot_accuracy` measured:

> final True/False accuracy at `ANS` while the ground-truth CoT prefix was already supplied.

In the parallel CoT format, matched pairs emit a tuple-index/letter token while unmatched pairs emit a sum digit. Consequently the supplied CoT contains a direct answer certificate. A model can therefore achieve perfect final answer accuracy without independently learning the pairwise computation.

New reports retain this quantity only as:

```text
cot_answer_given_cot_accuracy
```

and explicitly document that it is leakage-aware/non-causal evidence.

The informative intermediate diagnostics are:

```text
cot_pair_position_token_accuracy
cot_pair_position_semantic_accuracy
cot_sum_token_accuracy
cot_sum_semantic_accuracy
cot_match_index_accuracy
cot_result_semantic_accuracy
cot_result_nll
```

### Exact versus semantic metrics

With vocabulary reduction enabled, the target generator randomly chooses:

- one of the two pair-position labels; and
- one coordinate of a multidimensional pair sum.

Exact-token metrics intentionally mirror the sampled training target and therefore have stochastic ceilings.

Semantic metrics accept any target that is computationally equivalent for the same pair. They answer the more important diagnostic question:

> Did the model compute a valid pairwise result, independent of which equivalent reduced-vocabulary token happened to be sampled?

`cot_result_nll` measures teacher-forced negative log-likelihood over result slots only. It excludes pair-position tokens and the final answer.

## Phase 4 — Add training-answer visibility

Record answer accuracy on the training batches during the existing forward pass:

```text
epoch_train_answer_accuracies
best_train_answer_accuracy
```

This adds no second model forward.

The metric distinguishes two failure modes that training loss alone cannot separate:

```text
training answer remains near chance
    → optimization / wiring / capacity issue

training answer becomes high while validation remains chance
    → generalization / sample-complexity issue
```

## Phase 5 — Add explicit apparatus gates

### Gate A — structural/unit tests

CPU CI must prove:

- RoPE matches an independent pairwise rotation calculation;
- identical content at different positions receives different rotary transforms;
- the default filler budget remains exactly `n^2`;
- the separator is retained and supervised;
- every parallel-CoT diagnostic target belongs to its semantic-valid set;
- a scripted model can score 100% on final answer-given-CoT while scoring 0% on result semantics, proving the diagnostic distinguishes leakage from computation;
- RoPE/separator changes participate in deterministic run identity.

### Gate B — tiny fixed-set overfit

A very small length-3/dimension-1 immediate dataset is trained and evaluated on itself using a small CPU Llama.

Expected result:

```text
training answer accuracy >= 95%
validation-on-same-set accuracy >= 95%
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

This is a runtime diagnostic rather than a fixed CI threshold because generalization depends on optimization scale and should remain visible as experimental evidence rather than becoming a fragile unit test.

## Phase 6 — Re-run the replication progressively

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
training answer accuracy
filler validation accuracy
CoT pair/result generation diagnostics
CoT result NLL
majority baseline
```

Only after the apparatus produces a credible positive-control learning curve should Experiment 0A be considered validated and Experiment 0B/H2 interpretation resume.

## Interpretation rule

The pre-repair August 2026 Llama runs are diagnostic artifacts, not negative experimental results. They were executed with a positionless transformer and without a supervised continuation-boundary state, while the old CoT metric could saturate from the supplied CoT answer certificate.

The repaired protocol intentionally preserves the paper's `n^2` filler budget and dense parallel-CoT supervision while making the intermediate computation measurable.