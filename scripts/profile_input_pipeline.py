#!/usr/bin/env python3
"""Measure Experiment 0 input-pipeline capacity, and where its time goes.

Answers one question: can the loader keep the model fed? Compare the reported
capacity against the model's ``samples_per_second`` from a run report. If the
loader is comfortably ahead, a nonzero ``data_wait_fraction`` is worker startup,
not a throughput ceiling, and no tuning will help.

    python scripts/profile_input_pipeline.py
    python scripts/profile_input_pipeline.py --demand 1311 --batch-size 384
    python scripts/profile_input_pipeline.py --components

Runs on CPU only and never touches CUDA.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch.utils.data import DataLoader  # noqa: E402

from exp0.dataset import (  # noqa: E402
    Task3SumDataset,
    build_default_vocab,
    pad_collate_fn,
)
from exp0.generation import generate_protocol_packed_instances  # noqa: E402

DEFAULT_WORKER_COUNTS = (0, 2, 4, 8)


def build_dataset(
    samples: int,
    length: int,
    dimension: int,
    num_filler: Optional[int],
    seed: int,
    filler_only: bool,
) -> Task3SumDataset:
    vocab = build_default_vocab(length=length, dimension=dimension)
    packed = generate_protocol_packed_instances(
        samples, length=length, dimension=dimension, rng=random.Random(seed),
    )
    ratios: Dict[str, float] = {}
    if filler_only:
        ratios = {"parallel_ratio": 0.0, "filler_ratio": 1.0}
    return Task3SumDataset(
        packed,
        num_filler=length**2 if num_filler is None else num_filler,
        vocab=vocab,
        seed=seed,
        **ratios,
    )


def measure_capacity(
    dataset: Task3SumDataset,
    batch_size: int,
    workers: int,
    prefetch: int,
    pin_memory: bool,
) -> Dict[str, float]:
    """Return startup cost and steady-state capacity for one configuration.

    Capacity is derived from a whole second epoch, not from a median of
    inter-batch gaps. With prefetching most gaps are near zero while a few
    block, so a median reports capacities that are orders of magnitude too high.
    The second epoch also excludes worker startup, which persistent workers pay
    only once.
    """
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": pad_collate_fn,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch
    loader = DataLoader(**kwargs)

    start = time.perf_counter()
    first_batch_seconds = None
    for _ in loader:
        if first_batch_seconds is None:
            first_batch_seconds = time.perf_counter() - start
    second_start = time.perf_counter()
    for _ in loader:
        pass
    second_epoch_seconds = time.perf_counter() - second_start

    return {
        "workers": workers,
        "batch_size": batch_size,
        "prefetch_factor": prefetch if workers else 0,
        "pin_memory": pin_memory,
        "startup_seconds": first_batch_seconds or 0.0,
        "epoch_seconds": second_epoch_seconds,
        "capacity_samples_per_second": len(dataset) / second_epoch_seconds,
    }


def component_costs(dataset: Task3SumDataset, batch_size: int, samples: int) -> None:
    """Attribute per-sample cost inside __getitem__, and time the collate."""
    limit = min(samples, len(dataset))
    profiler = cProfile.Profile()
    profiler.enable()
    items = [dataset[index] for index in range(limit)]
    profiler.disable()

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(8)
    print("\ntop __getitem__ costs by tottime")
    for line in stream.getvalue().splitlines()[4:14]:
        if line.strip():
            print(f"  {line.strip()}")

    batch = items[:batch_size]
    start = time.perf_counter()
    repeats = 100
    for _ in range(repeats):
        pad_collate_fn(batch)
    collate_seconds = (time.perf_counter() - start) / repeats
    print(
        f"\n  pad_collate_fn({len(batch)}): {collate_seconds * 1000:.2f} ms/batch"
        f" -> {len(batch) / collate_seconds:,.0f} samples/s"
    )


def render(rows: Sequence[Dict[str, float]], demand: Optional[float]) -> None:
    print(f"\n  {'workers':>7} {'startup':>10} {'epoch':>9} {'capacity':>16}"
          f"{'  headroom' if demand else ''}")
    for row in rows:
        headroom = ""
        if demand:
            ratio = row["capacity_samples_per_second"] / demand
            headroom = f"  {ratio:8.1f}x"
        print(f"  {int(row['workers']):7d} {row['startup_seconds']:9.2f}s "
              f"{row['epoch_seconds']:8.2f}s "
              f"{row['capacity_samples_per_second']:12,.0f}/s{headroom}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--num-filler", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, nargs="+",
                        default=list(DEFAULT_WORKER_COUNTS))
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--filler-only", action="store_true",
                        help="single-format dataset; the CoT arm costs ~3.5x more")
    parser.add_argument("--demand", type=float, default=None,
                        help="model samples_per_second, to report headroom")
    parser.add_argument("--components", action="store_true",
                        help="attribute per-sample cost inside __getitem__")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    dataset = build_dataset(args.samples, args.length, args.dimension,
                            args.num_filler, args.seed, args.filler_only)
    print(f"dataset: {len(dataset)} samples, batch {args.batch_size}, "
          f"realized {dict(dataset.realized_counts)}")

    rows = [
        measure_capacity(dataset, args.batch_size, workers,
                         args.prefetch_factor, not args.no_pin_memory)
        for workers in args.workers
    ]
    render(rows, args.demand)

    if args.demand:
        best = max(rows, key=lambda row: row["capacity_samples_per_second"])
        startup = best["startup_seconds"]
        print(f"\n  A run whose epoch takes E seconds pays {startup:.1f}s of worker "
              f"startup once,\n  so data_wait_fraction is about {startup:.1f}/E. "
              "That shrinks as the run grows;\n  it is not a throughput problem "
              "and no tuning removes it.")

    if args.components:
        component_costs(dataset, args.batch_size, min(3000, len(dataset)))
    return 0


if __name__ == "__main__":
    # Required on Windows: DataLoader workers spawn by re-importing __main__, so
    # module-level dataset construction would re-run in every worker.
    raise SystemExit(main())
