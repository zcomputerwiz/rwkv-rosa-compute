# Experiment 0 Training Performance

This document describes implementation optimizations that reduce Experiment 0 CPU RAM, GPU memory, and training/evaluation overhead without changing the task definition.

## Default semantic-preserving optimizations

The runner now generates training and validation examples directly into compact tensor-backed storage instead of retaining one Python `Instance3Sum` object graph per sample.

For the default `length=12`, `dimension=3` task, packed instance storage uses:

```text
36 bytes  tuple digits: 12 * 3 * uint8
 1 byte   has_3sum: bool
 6 bytes  matching indices: 3 * int16
---------
43 bytes/sample
```

`Task3SumDataset` adds one `uint8` format code per sample, for 44 bytes/sample of packed instance/format backing storage. Therefore a 1,000,000-example default training set uses about 42 MiB for these tensors, excluding vocabulary, temporary batches, and Python/PyTorch object overhead.

The compact representation consumes the same RNG stream and reconstructs the same `Instance3Sum` values as the previous list-based generator. CPU tests compare the two representations directly. Generation fills NumPy buffers in place and wraps them once as Torch tensors, avoiding a tiny Torch allocation for every generated example.

Other default changes are intended to preserve the mathematical training/evaluation objective:

- `check_3sum` uses indexed complement lookup while retaining the previous `i -> j -> k` match ordering;
- training projects vocabulary logits only for positions used by the shifted next-token loss;
- evaluation projects the vocabulary head only at each example's `ANS` position;
- evaluation scoring is vectorized rather than synchronizing once per example;
- pinned CUDA DataLoader batches use non-blocking host-to-device copies;
- validation uses its own worker count, defaulting to zero, so training workers do not imply persistent validation workers;
- AdamW uses `zero_grad(set_to_none=True)`;
- the runner explicitly releases each trained model and per-seed training dataset before constructing the next seed, preventing accidental overlap between successive models in multi-seed runs.

The answer-only evaluation head performs a smaller matrix multiplication than the compatibility full-logit path. On some CPU/BLAS implementations this can change the last few floating-point bits, so regression tests require tight numerical agreement and identical argmax predictions rather than bitwise-equal logits.

## Mixed precision

The previous Experiment 0 protocol remains the default:

```text
--precision fp32
```

On CUDA GPUs with BF16 support, the recommended performance experiment is:

```text
--precision bf16
```

This uses PyTorch autocast for forward/loss/evaluation computation. BF16 is an explicit numerical protocol change and is recorded in `TrainConfig`, the report, and the deterministic run ID.

FP16 is also available:

```text
--precision fp16
```

FP16 training uses `torch.amp.GradScaler` and is likewise recorded as a distinct run configuration.

Non-FP32 precision currently requires CUDA. CPU runs should use FP32.

## Fused AdamW

PyTorch fused AdamW can be enabled explicitly on CUDA:

```text
--fused_adamw
```

It is disabled by default. Because fused optimizer kernels may change floating-point execution details, the setting participates in run identity.

## DataLoader tuning

Training and validation workers are independent:

```text
--num_workers 2
--val_num_workers 0
```

Validation defaults to zero workers because the fixed validation set is small and retaining additional persistent worker processes usually provides little benefit.

For training workers, the amount of queued data is controlled by:

```text
--prefetch_factor 2
```

On RAM-constrained Windows hosts, `--prefetch_factor 1` is a reasonable first comparison when `--num_workers` is nonzero.

Pinned memory is enabled by default and can be disabled with:

```text
--no-pin_memory
```

When pinning is active on CUDA, transfers use `non_blocking=True`.

## Suggested RTX 4060 Ti starting point

A reasonable first optimized smoke run is:

```bash
python scripts/run_experiment.py \
  --architecture llama \
  --device cuda \
  --precision bf16 \
  --batch_size 384 \
  --num_workers 2 \
  --val_num_workers 0 \
  --prefetch_factor 1 \
  --length 12 \
  --dimension 3 \
  --num_samples 20000 \
  --val_samples 2000 \
  --epochs 5 \
  --seeds 3 \
  --out_dir results/smoke
```

After establishing BF16 correctness for the intended GPU/software stack, `--fused_adamw` can be benchmarked separately so its effect is not conflated with the precision change.

## Instrumentation

Per-seed history records:

```text
precision
fused_adamw
non_blocking_transfers
train_dataset_storage_bytes
validation_dataset_storage_bytes
cuda_peak_memory_allocated_bytes
cuda_peak_memory_reserved_bytes
data_wait_seconds
epoch_seconds
samples_per_second
```

The CUDA peak-memory fields are `null` on CPU and are measured with PyTorch's CUDA allocator statistics on GPU. They make before/after VRAM comparisons reproducible from the report rather than relying only on an external `nvidia-smi` snapshot.

CPU CI verifies correctness/equivalence but does not claim CUDA speedups or VRAM savings. GPU throughput and peak allocated/reserved memory should be measured on the actual experiment machine before choosing a new production batch size.

## Deferred optimization: `torch.compile`

`torch.compile` is intentionally not enabled by this change. It can improve throughput on some model/GPU/software combinations, but compilation mode and CUDA graph behavior can also add startup cost or memory. It should be benchmarked separately on the actual experiment environment after the deterministic data/logit/AMP improvements above are established.
