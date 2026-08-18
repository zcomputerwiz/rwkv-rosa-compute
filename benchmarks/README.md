# ROSA Benchmarks

This directory contains benchmark scripts to evaluate execution latency across sequence lengths:

```text
T = 32, 64, 128, 256, 512
```

## Running Benchmarks

Run the standard benchmark (default 3 warmups, 10 repeats):

```bash
python benchmarks/benchmark_rosa.py
```

Run a quick smoke benchmark:

```bash
python benchmarks/benchmark_rosa.py --smoke
```

Customize warmups and measured iterations:

```bash
python benchmarks/benchmark_rosa.py --warmups 5 --repeats 20
```

## Methodology

- CPU benchmarks record precise `time.perf_counter()` timings across repeated iterations after initial warmups.
- CUDA benchmarks use `torch.cuda.Event` timing with `torch.cuda.synchronize()` before and after timing blocks.
- Results report mean and standard deviation (e.g. `mean ± std` in ms).
