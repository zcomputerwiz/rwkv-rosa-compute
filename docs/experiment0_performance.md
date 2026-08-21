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

## Compilation and precision levers (`torch.compile`, TF32, BF16)

Measured throughput levers and precision configurations are documented in detail in [`experiment0_precision_and_compile.md`](experiment0_precision_and_compile.md).

Key levers available via CLI flags:
- `--compile`: Enables `torch.compile(model.loss_logits)` with Triton codegen (delivers 1.40x on RWKV-7 and 2.12x on Llama combined with BF16 on Ada GPUs).
- `--tf32` / `--no-tf32`: Opt-in TensorFloat-32 for FP32 matmuls (~1.23x speedup on FP32 with minimal numerical deviation).
- `--precision bf16`: BF16 mixed precision autocast (1.61x speedup on Llama, standard for fused RWKV CUDA recurrence).

Reproduce and measure levers on your GPU using:
```bash
python scripts/benchmark_training_precision.py --arch both
```

On Windows, `torch.compile` requires the `triton-windows` package (installed via `pip install -e ".[cuda]"`) and MSVC toolset 14.44+ (handled automatically by `scripts/run_cuda_tests.ps1`).

## Sequence padding analysis & length-aware grouped execution

In mixed training batches (50% parallel CoT, 50% filler), batches are padded to the longest sequence in the batch. Because CoT sequences have fixed length (~40 tokens) regardless of $N$, filler examples at low $N$ (e.g. length 10 at $N=0$) waste substantial computation processing padding tokens.

To inspect and quantify padding waste across the $N$-sweep:
```bash
python scripts/analyze_sweep_padding.py --batch-size 128
```

At $N=0$, only ~51% of recurrent transitions are logical; the remaining 49% are padding and chunk-alignment overhead.

[`src/exp0/grouped_execution.py`](../src/exp0/grouped_execution.py) provides length-homogeneous subgroup execution. It partitions a mixed batch into homogeneous length groups, executes forward/backward passes per group with token-weighted loss summation (`reduction="sum"`), and applies a single global gradient clip and AdamW update per optimizer batch.

## RWKV-7 CUDA benchmark and profiling harness

`scripts/benchmark_rwkv_cuda.py` measures the fused recurrence independently of
the Experiment 0 training runner and also provides full randomly initialized
Experiment 0 RWKV-backbone modes. It never requires a checkpoint. Its JSON
artifact records schema version, provenance, matrix, timing statistics,
throughput, peak allocated/reserved memory, and per-configuration status.

### Cloud and CPU validation

CUDA is deliberately not needed to build and inspect a plan:

```bash
python scripts/benchmark_rwkv_cuda.py --dry-run --smoke
```

Add `--output results/cuda_benchmarks/smoke-plan.json` to serialize the plan. A
non-dry run on a CUDA-less host exits clearly rather than substituting CPU
timings or emitting invented results. Reference modes provide a same-device
comparison; they do not masquerade as a CUDA benchmark on CPU.

### Windows GPU runs

First validate a small matrix on either target GPU:

```powershell
python .\scripts\benchmark_rwkv_cuda.py `
  --smoke `
  --mode fused_forward `
  --mode fused_forward_backward `
  --output .\results\cuda_benchmarks\smoke.json
```

The standard recurrence matrix defaults to batches `1,2,4,8,16,32`, timesteps
`1,2,4,8,15,16,17,32,64,128`, hidden size 768, head dimension 64, 10 warmups,
and 50 iterations. Preserve a separate file from each target machine:

```powershell
python .\scripts\benchmark_rwkv_cuda.py `
  --mode fused_forward --mode fused_forward_backward `
  --mode reference_forward --mode reference_forward_backward `
  --warmups 10 --iterations 50 `
  --output .\results\cuda_benchmarks\recurrence.json
```

Measure full integration separately because it answers a different question:

```powershell
python .\scripts\benchmark_rwkv_cuda.py `
  --mode full_rwkv_forward --mode full_rwkv_forward_backward `
  --batches 1,2,4,8 --timesteps-list 16,17,64 `
  --output .\results\cuda_benchmarks\full-model.json
```

The standard matrix is 60 workloads per mode, so the four-mode command above is
240 workloads. The two reference modes alone execute roughly 200,000 sequential
Python-loop timestep iterations; budget hours, not minutes, and prefer an
explicit `--batches`/`--timesteps-list` subset when iterating.

OOM configurations are recorded and the matrix continues, as do ordinary Python
exceptions and configurations the fused kernel does not support. A device-side
assertion or other sticky CUDA fault is different in kind: it poisons the CUDA
context, so every later workload in the same process fails. Rerun the remaining
matrix in a fresh process rather than trusting results after one. Forward and forward+backward modes remain separate so saved-state and
gradient memory are visible. Before performance validation, use the existing
cold extension-build check, then benchmark in a fresh process:

```powershell
.\scripts\run_cuda_tests.ps1 -Cold
```

### Nsight-friendly single workload

Profiler mode requires one workload, warms it up, synchronizes at profiler
boundaries, and repeats it under a PyTorch NVTX range:

```powershell
python .\scripts\benchmark_rwkv_cuda.py `
  --mode fused_forward `
  --batch 16 --timesteps 64 `
  --profile --profile-iterations 100
```

Place that command after the desired `nsys profile` or `ncu` launcher. Nsight is
optional on the physical host and is not needed by cloud CI. Run normal mode
separately when a structured timing/memory JSON artifact is also required.

### Interpretation

- **Logical timesteps** are tokens requested by the caller.
- **Padded kernel timesteps** are the next multiple of `CHUNK_LEN=16` presented
  to the fused kernel (`1 -> 16`, `15 -> 16`, `16 -> 16`, `17 -> 32`).
- A recurrence transition is one head state update, so transition totals include
  batch and head counts. Physical totals use padded time.
- **Recurrence-only** isolates the recurrence. **Full-model** includes time-mix
  projections, channel mixing, normalization, and every configured layer.
- **Forward-only memory** is measured under `torch.no_grad()`, so it excludes
  both gradients and the activations autograd would otherwise save. Model
  parameters always require grad, so eval mode alone is not sufficient: without
  the no-grad guard, forward-only peaks include activations kept for a backward
  that never runs. **Forward+backward memory** exposes saved-state, gradient,
  and backward costs. Allocated and reserved peaks are distinct PyTorch
  allocator measurements.
- **Reference and fused modes differ in more than fusion.** `RWKV7_OP` is an
  FP32 Python loop over timesteps; the fused path is a bf16 CUDA kernel. Their
  ratio therefore combines kernel fusion, precision, and interpreter overhead,
  and must not be quoted as a fused-kernel speedup on its own.
- **The smallest cells measure launch overhead.** At `B=1, T=1` a single
  chunk-padded kernel launch dominates, so those rows characterize dispatch
  cost rather than recurrence throughput.
- **Unsupported configurations are distinct from failures.** The fused kernel
  requires `head_dim=64`; other values are recorded as `unsupported`, not
  `error`.

These benchmarks characterize implementation performance only. They are not
Experiment 0 accuracy results and must not be interpreted as evidence for or
against H1/H2. Padding structure alone is not evidence of a GPU bottleneck;
conclusions require measurements and profiler traces from the target systems.
