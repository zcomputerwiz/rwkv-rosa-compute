"""Profile where a 0B RWKV-7 training step actually spends GPU time.

Track I. The upstream-kernel audit assumed the fused recurrence was the thing
worth optimizing; the A/B showed it is roughly 8% of step time, so the question
"which upstream kernel should we adopt" is the wrong one until we know what the
other ~92% is.

This attributes self CUDA time to individual kernels and buckets them, for the
padded and grouped paths, with torch.compile on - the configuration a real run
would use. Compiled kernels lose module boundaries, so attribution is by kernel
name rather than by nn.Module.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from exp0.config import ModelConfig, Task3SumConfig
from exp0.dataset import Task3SumDataset, build_default_vocab, pad_collate_fn
from exp0.generation import generate_protocol_packed_instances
from exp0.grouped_execution import IGNORE_INDEX, grouped_loss_backward
from exp0.train import create_model

SHAPE = {"hidden": 768, "layers": 12, "heads": 12}

# Kernel-name buckets, checked in order; first match wins. Names come from the
# CUDA kernels themselves, so this is inherently a heuristic - the unmatched
# bucket is printed so nothing is silently swept into "other".
BUCKETS: List[Tuple[str, Tuple[str, ...]]] = [
    ("rwkv recurrence", ("forward_kernel", "backward_kernel")),
    ("matmul / gemm", ("gemm", "cutlass", "nvjet", "sm80_", "sm89_", "ampere_",
                       "s16816", "cublas", "dot_kernel", "addmm")),
    ("cross entropy", ("nll_loss", "log_softmax", "softmax", "cross_entropy")),
    ("optimizer", ("adam", "foreach", "multi_tensor", "amsgrad")),
    ("triton fused", ("triton_",)),
    ("elementwise / copy", ("elementwise_kernel", "vectorized_elementwise",
                            "copy_", "memcpy", "memset", "fill_", "cat_",
                            "index_", "gather", "scatter")),
    ("reduction / norm", ("reduce_kernel", "norm", "layer_norm", "group_norm",
                          "welford", "mean_kernel", "sum_kernel")),
]


def bucket_for(name: str) -> str:
    lowered = name.lower()
    for label, needles in BUCKETS:
        if any(needle in lowered for needle in needles):
            return label
    return "unclassified"


def build_batches(vocab, task_cfg, batch_size: int, num_filler: int, count: int):
    """Batches drawn from ONE dataset through a shuffled DataLoader.

    Building each batch as its own dataset gives every batch an exact 50/50
    split and therefore one constant pair of subgroup shapes. Real training
    shuffles a single dataset, so splits are binomial and produce ~18 distinct
    subgroup shape-sets per 100 batches - which is the condition CUDA graphs
    actually have to survive, since each distinct shape needs its own capture.
    """
    from torch.utils.data import DataLoader

    rng = random.Random(7)
    packed = generate_protocol_packed_instances(
        batch_size * count, length=task_cfg.length,
        dimension=task_cfg.dimension, rng=rng,
    )
    dataset = Task3SumDataset(
        packed, num_filler=num_filler, vocab=vocab,
        parallel_ratio=0.5, filler_ratio=0.5,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=pad_collate_fn,
        generator=torch.Generator().manual_seed(7),
        drop_last=True,
        # Matches train_cfg.pin_memory, which defaults True on CUDA. Without it
        # every host-to-device copy is synchronous regardless of non_blocking,
        # so an unpinned benchmark measures a transfer path training never uses.
        pin_memory=True,
    )
    return list(loader)


def make_model(vocab, task_cfg, device):
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    model = create_model(
        ModelConfig(
            architecture="rwkv",
            hidden_size=SHAPE["hidden"],
            num_hidden_layers=SHAPE["layers"],
            num_attention_heads=SHAPE["heads"],
            intermediate_size=SHAPE["hidden"] * 4,
            head_dim=64,
            rwkv_kernel="cuda",
            device="cuda",
            vocab_size=len(vocab),
            output_vocab_size=32000,
        ),
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab,
        task_cfg=task_cfg,
    )
    return model.to(device)


def padded_step(forward_fn, model, optimizer, batch, device,
                set_to_none: bool = True) -> None:
    optimizer.zero_grad(set_to_none=set_to_none)
    input_tuples = batch["input_tuples"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    loss_mask = batch["loss_mask"].to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = forward_fn(input_tuples, targets)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            loss_mask[:, 1:].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
    loss.backward()
    optimizer.step()


def grouped_step(forward_fn, model, optimizer, batch, device,
                 set_to_none: bool = True) -> None:
    optimizer.zero_grad(set_to_none=set_to_none)
    grouped_loss_backward(
        forward_fn, batch, device,
        autocast=lambda: torch.autocast("cuda", dtype=torch.bfloat16),
    )
    optimizer.step()


def profile_path(label: str, step_fn, batches, vocab, task_cfg, device,
                 warmup: int, steps: int,
                 record_shapes: bool = False,
                 fused_adamw: bool = False,
                 compile_mode: str = "default") -> Dict[str, Any]:
    model = make_model(vocab, task_cfg, device)
    forward_fn = torch.compile(
        model.loss_logits,
        **({} if compile_mode == "default" else {"mode": compile_mode}),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4,
        **({"fused": True} if fused_adamw else {}),
    )

    # CUDA graphs own the tensors they produce. Grouped execution runs one
    # backward per subgroup and accumulates into .grad, so freeing those buffers
    # each step (set_to_none=True) lets the second subgroup's backward overwrite
    # a graph-owned tensor from the first. Stable preallocated buffers, zeroed
    # rather than freed, are what the runtime asks for.
    graphed = compile_mode == "reduce-overhead"
    if graphed:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)

    for i in range(warmup):
        step_fn(forward_fn, model, optimizer, batches[i % len(batches)], device,
                set_to_none=not graphed)
    torch.cuda.synchronize()

    wall: List[float] = []
    for i in range(steps):
        start = time.perf_counter()
        step_fn(forward_fn, model, optimizer, batches[i % len(batches)], device,
                set_to_none=not graphed)
        torch.cuda.synchronize()
        wall.append(time.perf_counter() - start)

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=record_shapes) as prof:
        for i in range(steps):
            step_fn(forward_fn, model, optimizer, batches[i % len(batches)], device,
                    set_to_none=not graphed)
        torch.cuda.synchronize()

    per_kernel: Dict[str, float] = defaultdict(float)
    per_kernel_calls: Dict[str, int] = defaultdict(int)
    for event in prof.key_averages():
        cuda_us = getattr(event, "self_device_time_total", 0) or 0
        if cuda_us <= 0:
            continue
        per_kernel[event.key] += cuda_us
        per_kernel_calls[event.key] += event.count

    # NOTE: splitting the GEMM bucket into head vs backbone by operand shape
    # does not work. Kernel-level profiler events carry no input_shapes - only
    # aten op events do - so every kernel matches nothing and the head appears
    # to cost zero, which reads as a finding rather than as the measurement
    # failure it is. Isolating the head needs an ablation (vary
    # output_vocab_size and diff step time), not shape matching.
    head_us = 0.0
    gemm_us = 0.0

    total_us = sum(per_kernel.values())
    buckets: Dict[str, float] = defaultdict(float)
    for name, value in per_kernel.items():
        buckets[bucket_for(name)] += value

    del model, optimizer, forward_fn
    torch.cuda.empty_cache()
    return {
        "label": label,
        "wall_median_ms": statistics.median(wall) * 1e3,
        "head_gemm_us": head_us,
        "gemm_us_by_shape": gemm_us,
        "total_us": total_us,
        "steps": steps,
        "buckets": dict(buckets),
        "kernels": sorted(per_kernel.items(), key=lambda kv: -kv[1]),
        "calls": dict(per_kernel_calls),
    }


def report(result: Dict[str, Any], top: int) -> None:
    total = result["total_us"]
    steps = result["steps"]
    print(f"=== {result['label']} ===")
    gpu_ms = total / 1e3 / steps
    wall_ms = result.get("wall_median_ms", 0.0)
    print(f"  GPU time  {gpu_ms:.2f} ms/step")
    print(f"  wall time {wall_ms:.2f} ms/step"
          + (f"   gap {wall_ms - gpu_ms:+.2f} ms "
             f"({(wall_ms - gpu_ms) / wall_ms:.1%} not on the GPU)"
             if wall_ms else ""))
    print()
    print(f"  {'bucket':22} {'ms/step':>9} {'share':>8}")
    for name, value in sorted(result["buckets"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:22} {value / 1e3 / steps:>9.2f} {value / total:>7.1%}")
    print()
    if result.get("gemm_us_by_shape"):
        head = result["head_gemm_us"]
        gemm = result["gemm_us_by_shape"]
        print(f"  GEMM split by operand shape")
        print(f"    output head (32000)  {head / 1e3 / steps:>8.2f} ms/step "
              f"{head / total:>6.1%} of step")
        print(f"    backbone linears     {(gemm - head) / 1e3 / steps:>8.2f} ms/step "
              f"{(gemm - head) / total:>6.1%} of step")
        print()
    print(f"  top {top} kernels")
    print(f"  {'kernel':58} {'ms/step':>9} {'share':>7} {'calls':>7}")
    for name, value in result["kernels"][:top]:
        calls = result["calls"][name] // steps
        display = name if len(name) <= 56 else name[:53] + "..."
        print(f"  {display:58} {value / 1e3 / steps:>9.2f} "
              f"{value / total:>6.1%} {calls:>7}")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_filler", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--compile_mode",
                        choices=("default", "reduce-overhead", "max-autotune"),
                        default="default",
                        help="reduce-overhead uses CUDA graphs, which target "
                             "launch overhead rather than kernel time.")
    parser.add_argument("--fused_adamw", action="store_true",
                        help="Use fused AdamW. A numerical protocol change, "
                             "not a free speedup - see the grouped-execution doc.")
    parser.add_argument("--path", choices=("padded", "grouped", "both"),
                        default="both")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    task_cfg = Task3SumConfig(num_filler=args.num_filler)
    vocab = build_default_vocab(
        length=task_cfg.length, dimension=task_cfg.dimension, mod=task_cfg.mod
    )
    batches = build_batches(vocab, task_cfg, args.batch_size,
                            args.num_filler, args.batches)

    print(f"device     : {torch.cuda.get_device_name(0)}")
    print(f"config     : 0B RWKV-7, N={args.num_filler}, batch {args.batch_size}, "
          f"bf16 + compile"
          + (", fused AdamW" if args.fused_adamw else ", foreach AdamW"))
    print(f"padded     : B={batches[0]['targets'].shape[0]} "
          f"T={batches[0]['targets'].shape[1]}")
    print()

    paths = []
    if args.path in ("padded", "both"):
        paths.append(("padded", padded_step))
    if args.path in ("grouped", "both"):
        paths.append(("grouped", grouped_step))

    for label, step_fn in paths:
        result = profile_path(label, step_fn, batches, vocab, task_cfg,
                              device, args.warmup, args.steps,
                              fused_adamw=args.fused_adamw,
                              compile_mode=args.compile_mode)
        report(result, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
