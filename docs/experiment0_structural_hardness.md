# Experiment 0A structural-hardness contrast: N=0 vs N=36

Paired comparison of two Llama 0A arms on a frozen 6-stratum structural
challenge set. Both arms use identical protocol; the only difference is the
filler-token budget.

```text
challenge set   results/exp0_structural/structural_challenge_20260821.json
challenge_id    e06f92897411fe2e
content_sha256  bef50bba1c80600de6885bf60ef5f9fdfed1b37135715dce0b938cee6a1cb21b
instances       6000 (2000 positive / 4000 negative)
seeds           43, 44, 45 per arm
protocol        Llama 384h/4L, bf16, no compile, batch 384, 2M x 5 epochs
```

The `content_sha256` is a hash over the serialized instance records, not over
the file bytes, so it is transport-independent. It was recomputed
independently on both machines before any evaluation ran; all six evaluation
outputs record the same value.

## Headline

Every seed favours N=36, and the benefit is concentrated in the hard negatives.

```text
                     N=0      N=36     easy gain   hard gain   interaction
seed 43            0.7803   0.8080       +0.80%      +5.30%       +4.50%
seed 44            0.8312   1.0000       +0.30%     +46.70%      +46.40%
seed 45            0.6500   0.7372       +4.60%     +12.30%       +7.70%
```

"Interaction" is the hard-negative gain minus the easy-negative gain — the
degree to which filler tokens help *more* on structurally harder instances.

The strongest single result is seed-robust rather than statistical:

```text
instances every N=0 seed missed and every N=36 seed solved : 35
instances every N=0 seed solved and every N=36 seed missed :  0
```

That asymmetry requires agreement across six independently trained models, so
no single seed can produce it.

## Both arms are bimodal, and that is the finding

```text
        N=0 canonical    N=36 canonical
seed 43     84.35%           83.45%      (slightly WORSE with filler)
seed 44     86.90%          100.00%      (phase transition)
seed 45     70.55%           78.75%
```

Seed 44 at N=36 scores **100.00% on all six strata**, including
`corrupted_negative_near_3plus`, where every other model scores 21-53%. Its
training trajectory reached 0.996 after a single epoch. The same seed at N=0
climbed gradually to 0.870.

So the effect is not "filler tokens add a fixed increment". It is "filler
tokens sometimes enable a phase transition to actually computing the task, and
otherwise change relatively little". With three seeds and one transition, the
transition *rate* is not estimable.

**Do not lead with the mean interaction of +19.53%.** The three values are
4.50 / 46.40 / 7.70; the median is 7.70. The mean is carried almost entirely by
the seed that transitioned, and implies a typical effect four to five times
larger than two of three seeds show. Report per-seed, with the consistent
direction as the claim.

## The mechanism is rejection, not detection

In the two non-transitioning seeds the positive strata barely move, while the
negative strata move substantially:

```text
seed 43   positive_arm_positive             +1.60%   p = 2.27e-01
          corrupted_arm_surviving_positive  -0.10%   p = 1.00e+00
          corrupted_negative_near_2         +5.60%   p = 1.86e-03
          corrupted_negative_near_3plus     +5.30%   p = 3.96e-03

seed 45   positive_arm_positive             +1.40%   p = 3.38e-01
          corrupted_arm_surviving_positive  +0.40%   p = 8.12e-01
          corrupted_negative_near_1        +15.70%   p = 6.20e-25
          corrupted_negative_near_2        +17.90%   p = 1.89e-23
```

Filler tokens are helping the model **reject near-miss negatives**, not find
true positives. Supporting evidence: the True-prediction bias collapses from
+14.1 / +9.8 / +25.4 percentage points at N=0 to +4.8 / +0.0 / +7.9 at N=36.

The structural gradient itself is monotonic within every seed of both arms, so
it is not an artefact of that bias — a uniform bias would depress all four
negative strata equally rather than order them.

## Statistical cautions

**Do not pool per-instance data into a single McNemar test.** Pooling gives
p = 2.5e-197 over 18000 pairs, but those are three clusters of 6000 with
qualitatively different behaviour, and seed 44 alone supplies 39.9% of all
discordant pairs favouring N=36. A pooled test mostly measures "did any seed
transition" while presenting itself as instance-level significance. Per-seed
tests plus a seed-level summary is the honest form, and each seed is
individually significant anyway.

**No multiple-comparison correction is applied** across the 18 tests (6 strata
x 3 seeds). Most p-values survive any correction comfortably, but seed 43's
`near_0` (p = 7.68e-02) and `positive_arm_positive` (p = 2.27e-01) do not and
should not be described as trends.

**Both arms train on 5x less data than the earlier N=0 reference run**
(`ef1125605d565142`, 10M x 5 = 50M presentations, 99.30% accuracy). At 2M x 5
the arms sit near the phase transition, which is why seed variance is large. If
the transition rate is the quantity of interest, more seeds at this budget will
answer it better than more data per seed.

## Reproducing

```bash
python scripts/evaluate_structural_challenge.py \
  --checkpoint results/exp0_structural/n36/checkpoints/<run_id>/seed_43/epoch_005.pt \
  --challenge_set results/exp0_structural/structural_challenge_20260821.json \
  --precision bf16 --device cuda \
  --out results/exp0_structural/eval_n36_seed_43.json

python scripts/analyze_structural_experiment.py \
  --n0_evals  results/exp0_structural/eval_n0_seed_4{3,4,5}.json \
  --n36_evals results/exp0_structural/eval_n36_seed_4{3,4,5}.json \
  --out results/exp0_structural/structural_hardness_analysis.json
```

The N=36 arm is `run_id 6fae967b93ccdb6a`, seeds 43/44/45 in one invocation.
The N=0 arm was executed as three single-seed invocations with distinct
`run_id`s writing into a shared checkpoint directory. Since
`generate_protocol_packed_instances` is seeded per seed
(`rng=random.Random(seed)`), the resulting models are identical either way; the
difference is bookkeeping only, and the analysis consumes evaluation files
rather than training reports, so it is unaffected.
