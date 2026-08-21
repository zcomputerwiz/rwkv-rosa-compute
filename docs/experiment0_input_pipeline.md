# Experiment 0 input-pipeline capacity

How to tell whether the data loader is actually limiting a run, and what to do
if it is. The short answer for current configurations: it is not, and the
`data_wait_fraction` values seen on short probe runs are worker startup rather
than a throughput ceiling.

Reproduce any number here with:

```bash
python scripts/profile_input_pipeline.py --demand <model samples_per_second>
python scripts/profile_input_pipeline.py --components
```

## Read data_wait_fraction carefully

`data_wait_fraction` is the share of training time spent waiting on the loader.
It is a ratio, so it moves when *either* side moves. Two observations from the
same loader, same data, same worker count:

```text
rwkv_kernel=reference    29.8 samples/s    data_wait 0.020
rwkv_kernel=cuda        215.1 samples/s    data_wait 0.150
```

Making the model 7x faster raised the wait sevenfold. Nothing about the input
pipeline changed. A high value is therefore not by itself evidence of a loader
problem, and a low value is not evidence of a healthy one.

## Measured capacity

RTX 4060 Ti host, 16 logical cores, Windows, length 6 / dimension 3, 50/50
CoT/filler mixture, batch 128:

```text
workers   startup   capacity        vs 0B (280/s)   vs 0A (1,311/s)
      0     0.20s    1,580 /s            5.6x             1.2x
      2     3.58s    2,743 /s            9.8x             2.1x
      4     7.16s    4,415 /s           15.8x             3.4x
      8    14.88s    7,692 /s           27.5x             5.9x
```

Capacity scales at roughly 960 samples/s per worker. At the default
`--num_workers 2` the loader already runs between 2x and 10x ahead of every
model configuration measured so far.

### Capacity must come from a whole epoch

Do not compute capacity from a median of inter-batch gaps. With prefetching,
most gaps are near zero while a few block, so the median reports capacities that
are wrong by orders of magnitude — an early version of this measurement claimed
657,084 samples/s for an 8-worker loader whose real rate was 7,692. The script
times a complete second epoch instead, which also excludes the startup that
persistent workers pay only once.

## The startup explanation

Worker spin-up costs roughly 1.8s per worker on Windows, paid once per
`DataLoader`, because workers are spawned and re-import torch. Amortized against
epoch length it predicts every observed `data_wait_fraction`:

```text
run                startup / epoch      predicted    measured
10M x 5 epochs      3.58s / 7,628s        0.047%       0.08%
20k probe           3.58s / 71s            5.0%         7.9%
8k probe            3.58s / 37s            9.7%        15%
```

The quantity shrinks as the run grows and no tuning removes it. **At production
scale it is already 0.08%.**

## Where the per-sample cost is

`__getitem__` costs about 610 us/sample for the 50/50 mixture, and the CoT arm
dominates:

```text
_reduced_parallel_cot_tensors   77% of all __getitem__ time
  matching_k_after_pair         15 calls per CoT item (every (i,j) pair)
  _sum_token_entropy            ~14 calls per CoT item

parallel_cot item   ~1,110 us
filler item           ~322 us
pad_collate_fn(128)   3.4 ms/batch  ->  ~37,600 samples/s
pin_memory(128)       0.04 ms/batch
```

Collation and pinning are not worth attention: together they are a few percent
of the sample cost.

A note against a plausible-sounding wrong answer: the per-item
`random.Random(f"{seed}_{idx}")` construction looks expensive and is not. It is
4.7 us, 0.8% of the budget. It was measured before being blamed.

## Recommendations

1. **Change the flag, not the code.** Use `--num_workers 4` for long runs. It
   costs 7s of startup once and lifts capacity from 2,743 to 4,415 samples/s.
   This is insurance for configurations that raise model throughput, such as a
   move to bf16; it is not a fix for a present bottleneck.
2. **Leave `prefetch_factor` and `pin_memory` alone.** Both are measurably
   irrelevant here (epoch totals 4.23 / 4.32 / 4.42s for prefetch 2 / 4 / 8, and
   4.23 vs 4.31s for pinning on and off).
3. **Prefer single-format runs when the mixture is not the point.** A
   filler-only dataset is 3.7x cheaper per sample (5,839 vs 1,596 samples/s
   single-threaded), because the CoT diagnostics are what cost. This also
   removes the mixture caveat described in
   [`experiment0_positive_control_repair.md`](experiment0_positive_control_repair.md).
4. **Only then consider optimizing `_reduced_parallel_cot_tensors`.** It is the
   single hotspot, but it exists to produce diagnostics, and no measured
   configuration is currently limited by it.

## Windows note

`scripts/profile_input_pipeline.py` keeps all setup inside `main()` behind an
`if __name__ == "__main__"` guard. This is load-bearing, not style: DataLoader
workers spawn by re-importing `__main__`, so module-level dataset construction is
re-executed in every worker and the process deadlocks before producing output.
