# Where the RWKV 0B acquisition event lives

The 0B N=0 seed-42 run acquires hard-negative discrimination between epoch 3
and epoch 4: `corrupted_negative_near_3plus` AUC jumps 0.5201 -> 0.7120 (see
`docs/experiment0_0b_structural.md`). Both checkpoints are retained, so the
transition can be localized directly by patching one parameter group at a time
and re-evaluating on the frozen challenge set.

All figures are threshold-free AUC against the positive instances, measured on
`structural_challenge_20260821.json` (`content_sha256 bef50bba...`). Accuracy
is shown alongside but is not the measure of interest - several patches move
accuracy and AUC in opposite directions.

## Weight space does not show the event

Per-tensor relative change `||dw|| / ||w||` across each epoch boundary of the
same run:

```text
boundary   mean rel delta   top-20 share   top-50 share
1->2          0.20926          21.2%          42.7%
2->3          0.11584          19.0%          40.8%
3->4          0.06594          17.4%          39.8%   <- the acquisition
4->5          0.02411          18.2%          40.7%
```

The acquisition boundary has a **smaller** weight change than the two before
it, tracking the learning-rate decay, and its concentration is
indistinguishable from every non-transition boundary. Broken out per block or
per module type, no group is elevated at 3->4 relative to the other boundaries;
every ratio falls between 0.32x and 0.66x.

This is a null result about the instrument rather than about the effect. Block
11 moves most at *every* boundary, so parameter movement cannot separate "this
block always moves most" from "this block acquired the capability". Only a
functional test can.

## Necessity: reverting block 11 undoes 90% of the acquisition

Take the epoch-4 model, restore one group's epoch-3 weights, re-evaluate.
Reversion is the fraction of the 0.5201 -> 0.7120 span given back.

```text
config           chall  AUC all  near_3+  reversion
epoch 3 (pre)   48.87%   0.6497   0.5201         --
epoch 4 (post)  55.02%   0.7265   0.7120         --

ALL_blocks      49.40%   0.5543   0.4806     120.6%
block_11        63.07%   0.6342   0.5384      90.5%
head            41.93%   0.6814   0.6354      39.9%
block_09        55.10%   0.6434   0.6514      31.6%
block_10        52.63%   0.7046   0.6701      21.8%
block_00        56.70%   0.7075   0.6774      18.0%
block_07        55.43%   0.6664   0.6872      12.9%
block_08        51.92%   0.6762   0.6891      11.9%
block_06        54.45%   0.7207   0.6986       7.0%
block_04        52.32%   0.7025   0.7018       5.3%
input_proj      53.85%   0.7213   0.7050       3.6%
block_03        54.05%   0.7238   0.7064       2.9%
ln_out          55.37%   0.7236   0.7084       1.9%
block_05        54.65%   0.7280   0.7113       0.4%
block_01        54.97%   0.7214   0.7119       0.1%
block_02        53.23%   0.7183   0.7156      -1.9%
```

Reverting the final block alone returns hard-negative AUC to within 0.02 of the
pre-transition model. A depth gradient sits behind it - 11 far ahead of 10 and
9, then the head - while `input_proj` and the middle blocks do essentially
nothing.

Two details worth keeping:

- The `block_11` patch scores **63.07% challenge accuracy, above the unpatched
  epoch-4 model's 55.02%**, while its discrimination collapses. Accuracy and
  AUC move in opposite directions, which is why AUC is primary throughout.
- `ALL_blocks` reverts *past* the baseline (0.4806 against epoch 3's 0.5201).
  Reverting every block while keeping epoch-4's head and `ln_out` gives a model
  worse than either endpoint - the first sign of readout/representation
  mismatch, developed below.

## Sufficiency: no single block confers it, and block 11 makes it worse

Take the epoch-3 model, insert one group's epoch-4 weights, re-evaluate.

```text
config           chall  AUC all  near_3+   restored
epoch 3 (pre)   48.87%   0.6497   0.5201         --
epoch 4 (post)  55.02%   0.7265   0.7120         --

ALL_blocks      41.12%   0.6771   0.6333      59.0%
block_09        53.15%   0.6555   0.5424      11.6%
block_04        49.82%   0.6543   0.5359       8.2%
block_01        49.95%   0.6517   0.5299       5.1%
block_03        48.98%   0.6533   0.5274       3.8%
block_02        49.70%   0.6539   0.5233       1.7%
input_proj      51.55%   0.6495   0.5205       0.2%
ln_out          48.95%   0.6478   0.5193      -0.4%
block_06        49.15%   0.6464   0.5186      -0.8%
block_05        48.78%   0.6475   0.5155      -2.4%
block_08        56.37%   0.5611   0.5150      -2.6%
block_07        49.58%   0.6381   0.5135      -3.5%
block_00        44.65%   0.6304   0.5110      -4.7%
block_10        46.10%   0.6278   0.4973     -11.9%
block_11        38.03%   0.5479   0.4817     -20.0%
head            45.90%   0.5529   0.4779     -22.0%
```

The block whose removal cost 90% restores **-20.0%** when inserted on its own,
worse than doing nothing. The two most downstream groups, `block_11` and
`head`, are the two worst insertions in the table. The best single block is
block 9 at 11.6%, and all twelve together reach only 59%.

## Reading: a distributed representation with a late readout

Necessity is concentrated in block 11; sufficiency is concentrated nowhere. The
consistent explanation is that epoch 4 changed the computation across the whole
stack, and block 11 is where that change is converted into a decision. Removing
the readout hides the capability. Installing the readout without the
representation it was trained against is worse than leaving the model alone,
because the last block is then interpreting features never adapted to it.

The negative insertions carry this argument. If block 11 held an independent
decision rule, dropping it into the epoch-3 body would be neutral at worst.
Landing 20 points *below* baseline means the readout is tightly coupled to
upstream state.

## Consequence for block-wise training

**Freezing the body and training late blocks will not reproduce this
transition.** The measured mismatch penalty is large and negative, so a
schedule that updates a readout against a frozen representation works against
the mechanism rather than exploiting it.

This constrains block-wise schemes rather than ruling them out. A split that
keeps each readout with the representation it reads, or that re-adapts
downstream groups after upstream ones move, is untouched by this result.

## What this does not establish

- **One run, one seed, one transition.** This localizes the event in
  `c968fce9af66aa32` seed 42. Whether every transition localizes to the final
  block is unknown; the seed study is the way to find out.
- **Epoch granularity.** The patch endpoints are 62,500 steps apart, so "epoch
  4 changed the stack" covers a great deal of training. The event may be
  sharper than an epoch.
- **Per-group, not per-combination.** Interactions between groups are
  unmeasured apart from `ALL_blocks`.
- **No N=36 comparison.** Its checkpoints are on the other machine, and it
  never transitioned, so there is no matched event to localize.

## Reproducing

Both directions patch a group between `epoch_003.pt` and `epoch_004.pt` under
`results/exp0_0b_pilot/n0/checkpoints/c968fce9af66aa32/seed_42/`, write a
temporary checkpoint, and evaluate it with
`scripts/evaluate_structural_challenge.py` against
`results/exp0_structural/structural_challenge_20260821.json`. The evaluation
needs `ninja` and MSVC `cl.exe` on PATH for the JIT kernel; see
`scripts/run_cuda_tests.ps1`.
