"""Benchmark length-grouped execution against the padded path on GPU.

Answers four questions that the CPU parity work could not:

1. Does grouping actually reduce steady-state step time, and by how much?
2. Does the variable subgroup batch size cause torch.compile to recompile
   repeatedly during training? A one-off compile is irrelevant across millions
   of steps; a per-step recompile is fatal. Worse, exceeding dynamo's recompile
   limit silently falls back to eager, which would look like a slowdown with no
   error at all.
3. What fraction of head projections remain unsupervised after grouping? That
   decides whether a masked head projection is worth building or whether
   grouping already recovered the waste.
4. What (B, T) subgroups do real Experiment 0 batches actually produce?

Compile time and steady-state time are reported separately and never summed.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The padded path allocates a [B, T-1, 32000] fp32 logits tensor and its
# gradient; fragmentation rather than true capacity is what kills it first.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch._dynamo
import torch.nn.functional as F
from torch._dynamo.utils import counters

from exp0.config import ModelConfig, Task3SumConfig
from exp0.dataset import Task3SumDataset, build_default_vocab, pad_collate_fn
from exp0.generation import generate_protocol_packed_instances
from exp0.grouped_execution import (
    IGNORE_INDEX,
    group_by_length,
    grouped_loss_backward,
    supervised_token_count,
)
from exp0.train import create_model

# Experiment 0 0B shape.
RWKV_SHAPE = {"hidden": 768, "layers": 12, "heads": 12, "kernel": "cuda"}
LLAMA_SHAPE = {"hidden": 768, "layers": 12, "heads": 12, "kernel": "torch"}


def build_batches(
    vocab,
    task_cfg: Task3SumConfig,
    batch_size: int,
    num_filler: int,
    count: int,
    seed: int = 7,
) -> List[Dict[str, Any]]:
    """Real mixed 50/50 parallel-CoT / filler batches, left on the host.

    Drawn from ONE dataset through a shuffled DataLoader, the way training does.
    Building each batch as its own dataset instead gives every batch an exact
    50/50 split and therefore one constant pair of subgroup shapes, which makes
    any recompilation test pass trivially. Real batches are binomial around 50%
    and produce ~18 distinct subgroup shape-sets per 100 batches.

    grouped_loss_backward moves each subgroup itself, so batches stay on CPU and
    both paths pay the same transfer cost.
    """
    from torch.utils.data import DataLoader

    rng = random.Random(seed)
    packed = generate_protocol_packed_instances(
        batch_size * count, length=task_cfg.length,
        dimension=task_cfg.dimension, rng=rng,
    )
    dataset = Task3SumDataset(
        packed,
        num_filler=num_filler,
        vocab=vocab,
        parallel_ratio=0.5,
        filler_ratio=0.5,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
        # Matches train_cfg.pin_memory, which defaults True on CUDA.
        pin_memory=True,
    )
    return list(loader)


def make_model(arch: str, vocab, task_cfg: Task3SumConfig, device: torch.device):
    shape = RWKV_SHAPE if arch == "rwkv" else LLAMA_SHAPE
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    model = create_model(
        ModelConfig(
            architecture=arch,
            hidden_size=shape["hidden"],
            num_hidden_layers=shape["layers"],
            num_attention_heads=shape["heads"],
            intermediate_size=shape["hidden"] * 4,
            head_dim=64,
            rwkv_kernel=shape["kernel"],
            device="cuda",
            vocab_size=len(vocab),
            output_vocab_size=32000,
        ),
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab,
        task_cfg=task_cfg,
    )
    return model.to(device)


def padded_step(forward_fn: Callable, model, batch, device) -> Dict[str, Any]:
    """The current path: one rectangular forward/backward over the whole batch."""
    model.zero_grad(set_to_none=True)
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
    b, t = targets.shape
    head_positions = b * (t - 1)
    supervised = supervised_token_count(batch)
    return {
        "loss": float(loss.detach()),
        "head_positions": head_positions,
        "supervised_head_positions": supervised,
        "unsupervised_head_positions": head_positions - supervised,
    }


def grouped_step(forward_fn: Callable, model, batch, device) -> Dict[str, Any]:
    model.zero_grad(set_to_none=True)
    return grouped_loss_backward(
        forward_fn,
        batch,
        device,
        autocast=lambda: torch.autocast("cuda", dtype=torch.bfloat16),
    )


def compile_state() -> Dict[str, int]:
    """Snapshot of dynamo's compile bookkeeping."""
    import torch._dynamo.convert_frame as convert_frame

    return {
        "frames_compiled": int(convert_frame.FRAME_COUNTER),
        "graph_breaks": int(sum(counters["graph_break"].values())),
        "recompile_reasons": int(sum(counters["recompile_reasons"].values()))
        if "recompile_reasons" in counters
        else 0,
    }


def measure(
    label: str,
    step_fn: Callable,
    model,
    forward_fn: Callable,
    batches: List[Dict[str, Any]],
    device,
    warmup_steps: int,
    timed_steps: int,
) -> Dict[str, Any]:
    """Compile phase, recompile check, and steady-state timing, kept separate."""
    torch._dynamo.reset()
    counters.clear()
    before = compile_state()

    # --- Compile / warmup phase, timed separately and never added to the rest.
    torch.cuda.synchronize()
    warm_start = time.perf_counter()
    last_stats: Dict[str, Any] = {}
    for i in range(warmup_steps):
        last_stats = step_fn(forward_fn, model, batches[i % len(batches)], device)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - warm_start
    after_warmup = compile_state()

    # --- Steady-state recompile check. Any recompile after warmup raises here,
    # which is the failure mode that matters: a one-off compile is amortized
    # over millions of steps, a per-step recompile is not. Caught rather than
    # propagated so the benchmark still reports timings.
    torch._dynamo.config.error_on_recompile = True
    recompiled_after_warmup = False
    recompile_error = None
    try:
        for i in range(min(len(batches), 8)):
            step_fn(forward_fn, model, batches[i], device)
    except Exception as exc:  # dynamo raises a plain RuntimeError subclass
        recompiled_after_warmup = True
        recompile_error = str(exc).splitlines()[0][:200]
    finally:
        torch._dynamo.config.error_on_recompile = False

    # --- Steady-state timing.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    per_step: List[float] = []
    for i in range(timed_steps):
        start = time.perf_counter()
        last_stats = step_fn(forward_fn, model, batches[i % len(batches)], device)
        torch.cuda.synchronize()
        per_step.append(time.perf_counter() - start)

    after = compile_state()
    return {
        "label": label,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "compile_seconds": compile_seconds,
        "steady_median_ms": statistics.median(per_step) * 1e3,
        "steady_mean_ms": statistics.fmean(per_step) * 1e3,
        "steady_stdev_ms": statistics.stdev(per_step) * 1e3 if len(per_step) > 1 else 0.0,
        "frames_compiled_warmup": after_warmup["frames_compiled"] - before["frames_compiled"],
        "frames_compiled_total": after["frames_compiled"] - before["frames_compiled"],
        "graph_breaks": after["graph_breaks"],
        "recompiled_after_warmup": recompiled_after_warmup,
        "recompile_error": recompile_error,
        "loss": last_stats.get("loss"),
        "head_positions": last_stats.get("head_positions"),
        "unsupervised_head_positions": last_stats.get("unsupervised_head_positions"),
    }


def subgroup_distribution(batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The actual (B, T) subgroups real batches produce.

    This is what decides whether compile recompilation is bounded: a small set
    of distinct shapes recompiles a few times and then stops, while a long tail
    of unique shapes never converges.
    """
    shapes: Counter = Counter()
    for batch in batches:
        for length, index in group_by_length(batch):
            shapes[(int(index.numel()), int(length))] += 1
    return [
        {"batch": b, "target_length": t, "occurrences": n}
        for (b, t), n in sorted(shapes.items(), key=lambda kv: -kv[1])
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("rwkv", "llama"), default="rwkv")
    parser.add_argument("--num_filler", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Measure the eager baseline instead, for isolating compile effects.",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    task_cfg = Task3SumConfig(num_filler=args.num_filler)
    vocab = build_default_vocab(
        length=task_cfg.length, dimension=task_cfg.dimension, mod=task_cfg.mod
    )
    batches = build_batches(
        vocab, task_cfg, args.batch_size, args.num_filler, args.batches
    )

    print(f"device            : {torch.cuda.get_device_name(0)}")
    print(f"torch             : {torch.__version__}")
    print(f"arch / N          : {args.arch} / {args.num_filler}")
    print(f"batch_size        : {args.batch_size}")
    print(f"compile           : {not args.no_compile}")
    recompile_lim = getattr(
        torch._dynamo.config,
        "recompile_limit",
        getattr(torch._dynamo.config, "cache_size_limit", "N/A"),
    )
    print(f"dynamo recompile_limit : {recompile_lim}")
    print()

    print("subgroup (B, T) distribution over real batches")
    print(f"  {'B':>5} {'T':>5} {'count':>7}")
    for row in subgroup_distribution(batches):
        print(f"  {row['batch']:>5} {row['target_length']:>5} {row['occurrences']:>7}")
    padded = batches[0]["targets"].shape
    print(f"  padded rectangle: B={padded[0]} T={padded[1]}")
    print()

    results = []
    for label, step_fn in (("padded", padded_step), ("grouped", grouped_step)):
        model = make_model(args.arch, vocab, task_cfg, device)
        forward_fn = (
            model.loss_logits
            if args.no_compile
            else torch.compile(model.loss_logits)
        )
        results.append(
            measure(
                label,
                step_fn,
                model,
                forward_fn,
                batches,
                device,
                args.warmup,
                args.steps,
            )
        )
        del model
        torch.cuda.empty_cache()

    print("results")
    for r in results:
        print(f"  {r['label']}")
        print(f"    compile phase       : {r['compile_seconds']:.1f} s "
              f"({r['frames_compiled_warmup']} frames compiled)")
        print(f"    steady-state median : {r['steady_median_ms']:.2f} ms")
        print(f"    steady-state mean   : {r['steady_mean_ms']:.2f} "
              f"+/- {r['steady_stdev_ms']:.2f} ms")
        print(f"    frames compiled     : {r['frames_compiled_total']} total")
        print(f"    graph breaks        : {r['graph_breaks']}")
        print(f"    recompiles after warmup : {r['recompiled_after_warmup']}")
        if r["recompile_error"]:
            print(f"      {r['recompile_error']}")
        if r["head_positions"]:
            frac = 1.0 - r["unsupervised_head_positions"] / r["head_positions"]
            print(f"    head positions      : {r['head_positions']}, "
                  f"{frac:.1%} supervised")
        print(f"    peak memory         : {r['peak_memory_gib']:.2f} GiB")
        print(f"    loss                : {r['loss']:.6f}")

    base, grouped = results
    speedup = base["steady_median_ms"] / grouped["steady_median_ms"]
    print()
    print(f"steady-state speedup : {speedup:.3f}x "
          f"({base['steady_median_ms']:.2f} -> {grouped['steady_median_ms']:.2f} ms)")
    print(f"compile cost delta   : {grouped['compile_seconds'] - base['compile_seconds']:+.1f} s "
          "(one-off, excluded from the speedup)")
    if grouped["recompiled_after_warmup"]:
        print("WARNING: grouped path still recompiles after warmup. Over a long "
              "run this either burns compile time every step or exceeds the "
              "recompile limit and silently falls back to eager.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
