"""A/B the vendored RWKV-7 recurrence against the upstream v3 variants.

Track E. Compares three kernels at real Experiment 0 shapes:

    current   rwkv7_clampw                      (what we build today)
    v3        rwkv7_clampw_v3_for_h100          shared-memory preload of r,w,k,v,a,b
    v3_alt    rwkv7_clampw_v3_for_h100_alt      same, minus the v preload

Correctness is checked against the existing PyTorch reference recurrence
(RWKV7_OP) at the tolerances already used by tests/test_exp0_cuda.py. Those
tolerances are NOT relaxed to make a kernel pass; a variant that fails is
reported as failing.

One variant per process. Both v3 sources export the same cuda_forward_v3 /
cuda_backward_v3 symbols and share one binding that registers
TORCH_LIBRARY(rwkv7_clampw_v3), so loading two of them into a single
interpreter would collide on the operator namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

UPSTREAM = REPO_ROOT / "external" / "RWKV-LM" / "RWKV-v7" / "train_temp" / "cuda"

# Tolerances copied from tests/test_exp0_cuda.py. Do not loosen.
FORWARD_RTOL = FORWARD_ATOL = 8e-2
GRAD_RTOL = GRAD_ATOL = 8e-2

HEAD_DIM = 64
CHUNK_LEN = 16

VARIANTS: Dict[str, Dict[str, str]] = {
    "current": {"cu": "rwkv7_clampw.cu", "cpp": "rwkv7_clampw.cpp", "ns": "rwkv7_clampw"},
    "v3": {
        "cu": "rwkv7_clampw_v3_for_h100.cu",
        "cpp": "rwkv7_clampw_v3.cpp",
        "ns": "rwkv7_clampw_v3",
    },
    "v3_alt": {
        "cu": "rwkv7_clampw_v3_for_h100_alt.cu",
        "cpp": "rwkv7_clampw_v3.cpp",
        "ns": "rwkv7_clampw_v3",
    },
}


def load_variant(name: str):
    """Compile one variant and return its registered operator namespace."""
    import torch
    from torch.utils.cpp_extension import load

    spec = VARIANTS[name]
    flags = [
        "-res-usage",
        f"-D_N_={HEAD_DIM}",
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas=-O3",
        "--extra-device-vectorization",
    ]
    load(
        name=f"rwkv7_recurrence_ab_{name}",
        sources=[str(UPSTREAM / spec["cu"]), str(UPSTREAM / spec["cpp"])],
        is_python_module=False,
        verbose=False,
        extra_cuda_cflags=flags,
    )
    return getattr(torch.ops, spec["ns"])


def call_variant(ops, r, raw_w, k, v, a, b):
    """Forward through the raw operator, mirroring _RWKV7ClampW.forward.

    The kernel takes the PRE-softplus decay parameter and applies the transform
    internally; only the PyTorch oracle receives the transformed value. Tensors
    are 4D [B, T, H, N] and bf16, and T must already be chunk-aligned.
    """
    import torch

    tensors = [x.contiguous() for x in (r, raw_w, k, v, a, b)]
    batch, timesteps, heads, head_dim = tensors[0].shape
    out = torch.empty_like(tensors[3])
    state = torch.empty(
        batch, heads, timesteps // CHUNK_LEN, head_dim, head_dim,
        dtype=torch.float32, device=r.device,
    )
    sa = torch.empty(
        batch, timesteps, heads, head_dim,
        dtype=torch.float32, device=r.device,
    )
    ops.forward(*tensors, out, state, sa)
    return out


def reference(r, raw_w, k, v, a, b):
    import torch.nn.functional as F

    from exp0.models.rwkv import RWKV7_OP

    transformed_w = -F.softplus(-raw_w) - 0.5
    return RWKV7_OP(r, transformed_w, k, v, a, b, head_dim=HEAD_DIM)


def make_inputs(batch: int, timesteps: int, hidden: int, seed: int = 11):
    import torch

    torch.manual_seed(seed)
    shape = (batch, timesteps, hidden)
    scale = {"r": 0.1, "w": 1.0, "k": 0.1, "v": 0.1, "a": 0.1, "b": 0.1}
    return {
        key: (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * factor)
        for key, factor in scale.items()
    }


def run_variant(name: str, shapes: List[Dict[str, int]], steps: int) -> Dict[str, Any]:
    import torch

    ops = load_variant(name)
    result: Dict[str, Any] = {"variant": name, "shapes": []}

    for shape in shapes:
        raw = make_inputs(shape["batch"], shape["timesteps"], shape["hidden"])
        heads = shape["hidden"] // HEAD_DIM
        as_4d = {
            key: t.view(shape["batch"], shape["timesteps"], heads, HEAD_DIM)
            for key, t in raw.items()
        }

        # --- correctness against the PyTorch oracle, which takes 3D and the
        # transformed decay; the kernel takes 4D and the raw decay.
        ref_out = reference(
            raw["r"].float(), raw["w"].float(), raw["k"].float(),
            raw["v"].float(), raw["a"].float(), raw["b"].float(),
        )
        fused_out = call_variant(
            ops, as_4d["r"], as_4d["w"], as_4d["k"],
            as_4d["v"], as_4d["a"], as_4d["b"],
        ).reshape(shape["batch"], shape["timesteps"], shape["hidden"])
        try:
            torch.testing.assert_close(
                fused_out.float(), ref_out.float(),
                rtol=FORWARD_RTOL, atol=FORWARD_ATOL,
            )
            forward_ok, forward_err = True, None
        except AssertionError as exc:
            forward_ok = False
            forward_err = str(exc).splitlines()[0][:160]
        max_abs = float((fused_out.float() - ref_out.float()).abs().max())

        # --- timing, forward only (the backward needs the autograd wrapper)
        args_4d = (as_4d["r"], as_4d["w"], as_4d["k"],
                   as_4d["v"], as_4d["a"], as_4d["b"])
        for _ in range(3):
            call_variant(ops, *args_4d)
        torch.cuda.synchronize()
        per_step = []
        for _ in range(steps):
            start = time.perf_counter()
            call_variant(ops, *args_4d)
            torch.cuda.synchronize()
            per_step.append(time.perf_counter() - start)

        result["shapes"].append({
            **shape,
            "forward_ok": forward_ok,
            "forward_error": forward_err,
            "max_abs_deviation": max_abs,
            "median_ms": statistics.median(per_step) * 1e3,
            "stdev_ms": statistics.stdev(per_step) * 1e3 if len(per_step) > 1 else 0.0,
        })
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=None,
                        help="Internal: run exactly one variant in this process.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    # Real Experiment 0 subgroup shapes, from the grouped-execution benchmark.
    # T must be chunk-divisible; the CoT group is 136 -> 144, filler 4 -> 16.
    shapes = [
        {"batch": 24, "timesteps": 144, "hidden": 768, "label": "CoT group"},
        {"batch": 24, "timesteps": 16, "hidden": 768, "label": "filler group N=0"},
        {"batch": 48, "timesteps": 144, "hidden": 768, "label": "padded rectangle"},
    ]

    if args.variant:
        result = run_variant(args.variant, shapes, args.steps)
        print(json.dumps(result))
        return 0

    # Driver: one subprocess per variant, so operator namespaces cannot collide.
    results = []
    for name in ("current", "v3", "v3_alt"):
        proc = subprocess.run(
            [sys.executable, __file__, "--variant", name, "--steps", str(args.steps)],
            capture_output=True, text=True, env=os.environ.copy(),
        )
        if proc.returncode != 0:
            print(f"{name}: FAILED to build or run", file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)
            continue
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"tolerances: forward rtol/atol {FORWARD_RTOL} (from tests/test_exp0_cuda.py)")
    print()
    for shape_index, shape in enumerate(shapes):
        print(f"{shape['label']}  B={shape['batch']} T={shape['timesteps']} "
              f"C={shape['hidden']}")
        print(f"  {'variant':10} {'median ms':>10} {'stdev':>8} {'speedup':>8} "
              f"{'correct':>8} {'max dev':>10}")
        base = None
        for result in results:
            row = result["shapes"][shape_index]
            if base is None:
                base = row["median_ms"]
            print(f"  {result['variant']:10} {row['median_ms']:>10.3f} "
                  f"{row['stdev_ms']:>8.3f} {base / row['median_ms']:>7.3f}x "
                  f"{'PASS' if row['forward_ok'] else 'FAIL':>8} "
                  f"{row['max_abs_deviation']:>10.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
