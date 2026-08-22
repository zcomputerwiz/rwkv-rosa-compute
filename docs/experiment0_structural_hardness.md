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
**Read the next section before quoting any of this**: the same arm at 5x the
data reaches 96.4% on those hard negatives without filler tokens, so what
follows measures sample efficiency, not a capability gap.

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

## The effect is sample efficiency, not capability separation

The obvious question this design cannot answer on its own is whether N=0 simply
needs more data. It is answerable for free: `ef1125605d565142` is a completed
N=0 run at 10M x 5 (5x this experiment's budget) with resumable checkpoints, and
evaluating it on the frozen challenge set is inference over 6000 instances.

```text
corrupted_negative_near_3plus (the hard negatives)
  N=0  @  2M x 5     42.3% / 53.3% / 21.3%    mean 38.97%
  N=36 @  2M x 5     47.6% / 100.0% / 33.6%   mean 60.50%
  N=0  @ 10M x 5     96.40%                   same arm, 5x data
```

**N=0 at full budget nearly solves the hard negatives**, beating two of the
three N=36 seeds at 2M and approaching the transitioned seed's 100%. Its
canonical accuracy reproduces the recorded 99.30% exactly, so the evaluation is
sound.

Full stratum profile of the 10M reference:

```text
positive_arm_positive             98.80%
corrupted_arm_surviving_positive  98.40%
corrupted_negative_near_0        100.00%
corrupted_negative_near_1         99.90%
corrupted_negative_near_2         99.60%
corrupted_negative_near_3plus     96.40%
True-prediction bias               +0.2%
```

So filler tokens **accelerate convergence** rather than unlocking a capability
N=0 cannot reach. The structural gradient reported below is a property of
undertrained models, and it flattens as training proceeds regardless of whether
the extra compute comes from filler tokens or from more data.

That is a real result, but a weaker one than "filler tokens enable hard-instance
reasoning". It also changes what an N-sweep would measure: the quantity of
interest is the sample efficiency of each N, not an asymptotic capability
difference, and every arm needs enough budget for its own convergence before the
comparison means anything.

Caveats on this single run: one seed (42, not 43/45/45), fp32 rather than bf16,
and n=1. The 96.4% versus 21-53% gap is far outside the seed variance seen
anywhere else here, but a matched 10M N=36 arm is what would close the argument
properly.

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
true positives.

The obvious objection is that this is just a decision-threshold shift: predict
False more often, gain on negatives, lose on positives. It is not, and the
cleanest disproof needs no appeal to stratum monotonicity. On the challenge set
the True-prediction bias does move toward False, and positive accuracy rises
anyway:

```text
          True-bias  N=0 -> N=36     positive_arm_positive
seed 43      +14.1%  ->  +11.8%             +1.6%
seed 45      +25.4%  ->  +17.2%             +1.4%
```

A pure threshold shift toward False **must** lower positive accuracy, because
fewer positives are predicted True. Positives went up in both seeds. The model
is discriminating better, not deciding differently.

The 10M N=0 reference is the same phenomenon at its endpoint: bias +0.2% with
98.8% on positives. Calibration and discrimination improve together as training
proceeds — which is consistent with the sample-efficiency reading above rather
than with filler tokens supplying a distinct mechanism.

(Both bias figures above are measured on the challenge set. An earlier revision
quoted canonical-validation biases alongside challenge-set accuracies; the
argument holds either way but the numbers must come from one measurement.)

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
(`ef1125605d565142`, 10M x 5 = 50M presentations, 99.30% accuracy). That gap is
now resolved rather than merely flagged — see the sample-efficiency section: the
reference clears the hard negatives at 96.4% with no filler tokens at all. At
2M x 5 both arms sit near the phase transition, which is why seed variance is
large. If the transition rate is the quantity of interest, more seeds at this
budget answer it better than more data per seed; if the asymptotic difference
is, a matched 10M N=36 arm is required and this experiment does not address it.

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
