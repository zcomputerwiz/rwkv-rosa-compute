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


def compare():
    info = get_environment_info()
    print("=== ROSA Diagnostic Comparison ===")
    print(f"PyTorch Version: {info['torch_version']}")
    print(f"CUDA Available:  {info['cuda_available']}")

    B, T, C = 1, 32, 768
    torch.manual_seed(42)
    q = torch.randn(B, T, C)
    k = torch.randn(B, T, C)
    v = torch.randn(B, T, C)

    print(f"\nComparing implementations on tensor shape: [B={B}, T={T}, C={C}]")

    # 1. BlinkDL reference
    t0 = time.perf_counter()
    out_blinkdl = blinkdl_rosa_4bit_reference(q, k, v)
    t_blinkdl = (time.perf_counter() - t0) * 1000

    # 2. rosa_soft reference
    t0 = time.perf_counter()
    out_rosa_soft_ref = rosa_4bit_forward(q, k, v, max_suffix_length=512, use_cuda=False)
    t_rosa_soft_ref = (time.perf_counter() - t0) * 1000

    max_diff = (out_blinkdl - out_rosa_soft_ref).abs().max().item()
    mean_diff = (out_blinkdl - out_rosa_soft_ref).abs().mean().item()
    exact_match_count = (out_blinkdl == out_rosa_soft_ref).sum().item()
    total_elements = out_blinkdl.numel()

    print(f"BlinkDL Reference Time:       {t_blinkdl:.2f} ms")
    print(f"rosa_soft Reference Time:     {t_rosa_soft_ref:.2f} ms")
    print(f"Max Absolute Difference:      {max_diff:.6f}")
    print(f"Mean Absolute Difference:     {mean_diff:.6f}")
    print(f"Exact Equality Count:         {exact_match_count} / {total_elements} ({exact_match_count/total_elements:.2%})")

    # 3. rosa_soft CUDA if available
    if info['cuda_available'] and info.get('rosa_soft_build_capabilities') and info['rosa_soft_build_capabilities'].rosa_soft_cuda:
        q_cuda = q.cuda()
        k_cuda = k.cuda()
        v_cuda = v.cuda()

        # Warmup
        _ = rosa_4bit_forward(q_cuda, k_cuda, v_cuda, max_suffix_length=512, use_cuda=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        out_cuda = rosa_4bit_forward(q_cuda, k_cuda, v_cuda, max_suffix_length=512, use_cuda=True)
        torch.cuda.synchronize()
        t_cuda = (time.perf_counter() - t0) * 1000

        diff_cuda = (out_rosa_soft_ref - out_cuda.cpu()).abs().max().item()
        print(f"rosa_soft CUDA Time:          {t_cuda:.2f} ms")
        print(f"CUDA vs Ref Max Abs Diff:     {diff_cuda:.6f}")
    else:
        print("rosa_soft CUDA:               Not Available / Skipped")

if __name__ == "__main__":
    compare()
