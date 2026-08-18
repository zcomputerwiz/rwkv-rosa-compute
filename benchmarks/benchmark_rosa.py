#!/usr/bin/env python3
import argparse
import statistics
import time

import torch

from rosa_compute import (
    blinkdl_rosa_4bit_reference,
    get_environment_info,
    rosa_4bit_forward,
)


def benchmark_cpu_fn(fn, warmups: int, repeats: int) -> tuple[float, float]:
    """Runs CPU timing loops with warmup and returns (mean_ms, std_ms)."""
    for _ in range(warmups):
        _ = fn()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std


def benchmark_cuda_fn(fn, warmups: int, repeats: int) -> tuple[float, float]:
    """Runs CUDA timing loops with events and synchronization, returning (mean_ms, std_ms)."""
    for _ in range(warmups):
        _ = fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        _ = fn()
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std


def run_benchmarks():
    parser = argparse.ArgumentParser(description="ROSA Compute Latency Benchmark")
    parser.add_argument("--warmups", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=10, help="Measured iterations")
    parser.add_argument(
        "--smoke", action="store_true", help="Run fast smoke test with fewer iterations and short sequences"
    )
    args = parser.parse_args()

    if args.smoke:
        warmups = 1
        repeats = 2
        sequence_lengths = [32, 64]
    else:
        warmups = args.warmups
        repeats = args.repeats
        sequence_lengths = [32, 64, 128, 256, 512]

    info = get_environment_info()
    print("=== ROSA Execution Latency Benchmark ===")
    print(f"PyTorch Version: {info['torch_version']}")
    print(f"CUDA Available:  {info['cuda_available']}")
    print(f"Warmup Iters:    {warmups}")
    print(f"Repeat Iters:    {repeats}")

    B = 1
    C = 768

    print(
        f"\n{'T':<6} | {'BlinkDL Ref (ms)':<22} | {'rosa_soft Ref (ms)':<22} | {'rosa_soft CUDA (ms)':<22}"
    )
    print("-" * 80)

    caps = info.get("rosa_soft_build_capabilities")
    has_cuda = info["cuda_available"] and caps and caps.rosa_soft_cuda

    for T in sequence_lengths:
        torch.manual_seed(42)
        q = torch.randn(B, T, C)
        k = torch.randn(B, T, C)
        v = torch.randn(B, T, C)

        # BlinkDL ref (CPU oracle)
        if T <= 128:
            m_b, s_b = benchmark_cpu_fn(
                lambda: blinkdl_rosa_4bit_reference(q, k, v), warmups, repeats
            )
            str_blinkdl = f"{m_b:.2f} ± {s_b:.2f}"
        else:
            str_blinkdl = "skipped (slow)"

        # rosa_soft ref
        m_r, s_r = benchmark_cpu_fn(
            lambda: rosa_4bit_forward(q, k, v, max_suffix_length=512, use_cuda=False),
            warmups,
            repeats,
        )
        str_soft_ref = f"{m_r:.2f} ± {s_r:.2f}"

        # rosa_soft CUDA
        if has_cuda:
            q_cuda, k_cuda, v_cuda = q.cuda(), k.cuda(), v.cuda()
            m_c, s_c = benchmark_cuda_fn(
                lambda: rosa_4bit_forward(q_cuda, k_cuda, v_cuda, max_suffix_length=512, use_cuda=True),
                warmups,
                repeats,
            )
            str_cuda = f"{m_c:.2f} ± {s_c:.2f}"
        else:
            str_cuda = "N/A"

        print(f"{T:<6} | {str_blinkdl:<22} | {str_soft_ref:<22} | {str_cuda:<22}")


if __name__ == "__main__":
    run_benchmarks()
