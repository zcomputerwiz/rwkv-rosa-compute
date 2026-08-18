#!/usr/bin/env python3
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rosa_compute import (
    blinkdl_rosa_4bit_reference,
    get_environment_info,
    rosa_4bit_forward,
)


def run_benchmarks():
    info = get_environment_info()
    print("=== ROSA Execution Latency Benchmark ===")
    print(f"PyTorch Version: {info['torch_version']}")
    print(f"CUDA Available:  {info['cuda_available']}")

    B = 1
    C = 768
    sequence_lengths = [32, 64, 128, 256, 512]

    print(f"\n{'T':<8} | {'BlinkDL Ref (ms)':<18} | {'rosa_soft Ref (ms)':<20} | {'rosa_soft CUDA (ms)':<20}")
    print("-" * 75)

    caps = info.get("rosa_soft_build_capabilities")
    has_cuda = info["cuda_available"] and caps and caps.rosa_soft_cuda

    for T in sequence_lengths:
        torch.manual_seed(42)
        q = torch.randn(B, T, C)
        k = torch.randn(B, T, C)
        v = torch.randn(B, T, C)

        # BlinkDL ref (CPU oracle)
        if T <= 128:
            t0 = time.perf_counter()
            _ = blinkdl_rosa_4bit_reference(q, k, v)
            t_blinkdl = (time.perf_counter() - t0) * 1000
            s_blinkdl = f"{t_blinkdl:.2f}"
        else:
            s_blinkdl = "skipped (slow)"

        # rosa_soft ref
        t0 = time.perf_counter()
        _ = rosa_4bit_forward(q, k, v, max_suffix_length=512, use_cuda=False)
        t_soft_ref = (time.perf_counter() - t0) * 1000
        s_soft_ref = f"{t_soft_ref:.2f}"

        # rosa_soft CUDA
        if has_cuda:
            q_cuda, k_cuda, v_cuda = q.cuda(), k.cuda(), v.cuda()
            _ = rosa_4bit_forward(q_cuda, k_cuda, v_cuda, max_suffix_length=512, use_cuda=True)
            torch.cuda.synchronize()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            _ = rosa_4bit_forward(q_cuda, k_cuda, v_cuda, max_suffix_length=512, use_cuda=True)
            end_evt.record()
            torch.cuda.synchronize()
            t_cuda = start_evt.elapsed_time(end_evt)
            s_cuda = f"{t_cuda:.2f}"
        else:
            s_cuda = "N/A"

        print(f"{T:<8} | {s_blinkdl:<18} | {s_soft_ref:<20} | {s_cuda:<20}")


if __name__ == "__main__":
    run_benchmarks()
