# Experiment 0B: RWKV-7 N=0 on the structural challenge set

The 0B RWKV-7 N=0 arm evaluated on the same frozen 6-stratum challenge set used
for the 0A Llama arms. This is the control half of a pair; the matching N=36 arm
runs on the Ampere laptop and is not yet available, so nothing here speaks to
the filler-token hypothesis on its own.

```text
challenge set   results/exp0_structural/structural_challenge_20260821.json
challenge_id    e06f92897411fe2e
content_sha256  bef50bba1c80600de6885bf60ef5f9fdfed1b37135715dce0b938cee6a1cb21b
evaluation      results/exp0_0b_structural/eval_rwkv_n0_seed_42.json
run_id          c968fce9af66aa32, seed 42
protocol        RWKV-7 768h/12L head_dim 64, bf16, compile + grouped + fused
                AdamW, batch 32, 2M x 5 epochs, no early stop triggered
```

The `content_sha256` matches the value recorded by all six Llama evaluations,
so this is the identical instance set and the comparison below is like-for-like.

## Results

```text
arm                  canon   chall    bias      pos corr_pos  near_0  near_1  near_2 near_3+
Llama N=0  s43      84.35%  78.03%  +14.1%   88.30%  88.00%  98.60%  86.30%  64.70%  42.30%
Llama N=0  s44      86.90%  83.12%   +9.8%   90.50%  88.20%  99.70%  90.40%  76.60%  53.30%
Llama N=0  s45      70.55%  65.00%  +25.4%   84.90%  86.20%  93.70%  64.90%  39.00%  21.30%
Llama N=36 s43      83.45%  80.80%  +11.8%   89.90%  87.90%  99.40%  89.70%  70.30%  47.60%
Llama N=36 s44     100.00% 100.00%   +0.0%  100.00% 100.00% 100.00% 100.00% 100.00% 100.00%
Llama N=36 s45      78.75%  73.72%  +17.2%   86.30%  86.60%  98.30%  80.60%  56.90%  33.60%
Llama N=0 10M       99.30%  98.85%   +0.2%   98.80%  98.40% 100.00%  99.90%  99.60%  96.40%
RWKV  N=0  s42      66.10%  60.60%  +27.0%   81.60%  81.30%  64.90%  48.20%  43.10%  44.50%
```

`bias` is the challenge-set True-prediction bias: predicted-true minus
actual-true, as a percentage of the 6000 instances.

## The negative-stratum profile is flat, and that is the finding

Every Llama arm rejects the structurally easy negatives almost perfectly and
degrades as near-matches accumulate. The RWKV arm does not have that gradient:

```text
                 near_0   near_1   near_2   near_3+   spread
Llama N=0 s43    98.60%   86.30%   64.70%   42.30%    56.3 points
Llama N=0 s44    99.70%   90.40%   76.60%   53.30%    46.4 points
Llama N=0 s45    93.70%   64.90%   39.00%   21.30%    72.4 points
RWKV  N=0 s42    64.90%   48.20%   43.10%   44.50%    21.8 points
```

`corrupted_negative_near_0` instances have no near-matches at all and are the
easiest negatives in the set. Llama scores 93.7-99.7% on them. RWKV scores
64.9%, then flattens to 43-48% across the three harder strata - close to chance
on a balanced binary decision, and essentially independent of how hard the
instance is.

Combined with the +27.0% True-bias and 81% on both positive strata, the reading
is that this model has learned to answer True by default and is near-guessing
whenever the answer is False. It is not doing graded rejection at all, whereas
the Llama arms at the same sample budget already are and fail only at the hard
end.

Note that RWKV's `near_3plus` (44.50%) is **higher** than two of the three Llama
N=0 seeds. That is not evidence of strength. It is a flat profile crossing a
descending one, and quoting that single cell without the rest of the row would
invert the conclusion.

Nearly half the decisions are marginal: 46.6% of instances have an absolute
logit margin below 0.5 (median 0.500). The model is not confidently wrong, it
is undecided.

## What this cannot support

- **n=1.** One seed against two or three per Llama arm. The Llama N=36 seeds
  spanned 0.788 to 1.000 on filler accuracy, so a single seed cannot separate
  "architecture is behind" from "this seed landed low".
- **The optimization settings are not matched to the Llama arms.** Both use
  `lr 1e-4` with the same schedule, but RWKV runs at batch 32 against Llama's
  384. With no gradient accumulation that is 12x as many updates at the same
  step size, so the effective per-sample learning rate differs by roughly an
  order of magnitude. "RWKV learns this task more slowly than Llama" is
  therefore **not** supported by this comparison; the batch and LR would have to
  be matched, or scaled by the usual linear/sqrt heuristics, before that claim
  could be made.
- **Undertrained, by design.** The 0A work established that N=0 at 5x this
  budget reaches 96.4% on the hard negatives, so a weak N=0 result at 2M x 5 is
  the expected control behaviour and not a capability ceiling.

## What the paired N=36 arm should show

If the filler-token effect replicates in RWKV, the N=36 arm should show a
*shape* change and not merely a higher number: a recovered gradient across the
negative strata, driven by `near_0` moving toward the 90s, and a True-bias
falling well below +27.0%. A uniform lift with the profile still flat would
indicate a better-calibrated guesser rather than a model that has started
computing the task.

## Reproducing

The evaluation needs the RWKV CUDA kernel, which is JIT-compiled and therefore
needs `ninja` and MSVC `cl.exe` on PATH. Invoking `.venv\Scripts\python.exe`
directly does not provide either: `ninja.exe` lives in `.venv\Scripts`, which is
only on PATH when the venv is activated, and `cl.exe` needs `vcvars64.bat`.
`scripts/run_cuda_tests.ps1` contains the supported setup (vswhere, newest
toolset, environment imported into the session); reuse it rather than
hand-rolling, or the kernel load fails with a bare
"Failed to compile/load the RWKV-7 CUDA kernel".

```bash
python scripts/evaluate_structural_challenge.py \
  --checkpoint results/exp0_0b_pilot/n0/checkpoints/c968fce9af66aa32/seed_42/epoch_005.pt \
  --challenge_set results/exp0_structural/structural_challenge_20260821.json \
  --precision bf16 --device cuda --batch_size 64 \
  --out results/exp0_0b_structural/eval_rwkv_n0_seed_42.json
```
