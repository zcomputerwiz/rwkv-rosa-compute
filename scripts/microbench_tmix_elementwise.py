"""Isolate the RWKV-7 TimeMix elementwise block for per-kernel profiling.

The step profile puts ~41% of GPU time in elementwise and triton-fused kernels.
Whether that is near its ceiling depends on which ceiling: the tensors are about
5 MiB each and this card has 32 MiB of L2, so much of the traffic may be
L2-served at roughly 1.2 TB/s rather than DRAM-served at 288 GB/s. Comparing
achieved throughput against the DRAM peak therefore flatters the result.

Small enough to run under Nsight Compute without profiling the whole model.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F


def tmix_elementwise(x, xx, mus, k_k, k_a, a, heads):
    """The pointwise surface of RWKV7TimeMix.forward, without the projections."""
    xr, xw, xk, xv, xa, xg = (x + xx * mu for mu in mus)
    b, t, c = x.shape
    kk = xk * k_k
    kk = F.normalize(kk.view(b, t, heads, -1), dim=-1, p=2.0).view(b, t, c)
    k = xk * (1 + (a - 1) * k_a)
    return xr, xw, kk, k, xv, xg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=3360)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    dev = torch.device("cuda")
    shape = (1, args.tokens, args.channels)
    x = torch.randn(shape, device=dev, dtype=torch.bfloat16)
    xx = torch.randn(shape, device=dev, dtype=torch.bfloat16)
    a = torch.rand(shape, device=dev, dtype=torch.bfloat16)
    mus = [torch.randn(1, 1, args.channels, device=dev, dtype=torch.bfloat16)
           for _ in range(6)]
    k_k = torch.randn(1, 1, args.channels, device=dev, dtype=torch.bfloat16)
    k_a = torch.randn(1, 1, args.channels, device=dev, dtype=torch.bfloat16)

    fn = tmix_elementwise
    if args.compile:
        fn = torch.compile(tmix_elementwise)

    for _ in range(10):
        fn(x, xx, mus, k_k, k_a, a, args.heads)
    torch.cuda.synchronize()

    per = []
    for _ in range(args.steps):
        s = time.perf_counter()
        fn(x, xx, mus, k_k, k_a, a, args.heads)
        torch.cuda.synchronize()
        per.append(time.perf_counter() - s)

    ms = statistics.median(per) * 1e3
    tensor_mb = args.tokens * args.channels * 2 / 1e6
    # 2 reads (x, xx) + 6 lerp writes + normalize read/write + k read/write
    passes = 12
    gb = passes * tensor_mb / 1e3
    print(f"tokens {args.tokens}  channels {args.channels}  compile {args.compile}")
    print(f"  one tensor        {tensor_mb:8.2f} MB   (L2 is 32 MiB)")
    print(f"  median            {ms:8.3f} ms")
    print(f"  est traffic       {gb:8.3f} GB  ({passes} passes)")
    print(f"  implied BW        {gb / (ms / 1e3):8.0f} GB/s")
    print(f"  vs DRAM peak 288  {gb / (ms / 1e3) / 288:8.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
