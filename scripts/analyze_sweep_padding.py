#!/usr/bin/env python3
"""Quantify padding waste in the Experiment 0 N sweep.

The mixed 50/50 CoT/filler batch is padded to its longest member. Parallel CoT
is a fixed length regardless of N, so every low-N filler example is carried
through a rectangle sized for CoT, and the fused recurrence then rounds time up
again to CHUNK_LEN. A filler example whose logical length is 10 can execute at
48 physical recurrence positions.

That waste is an implementation artifact, not part of the experiment. The
scientific compute budget remains the requested filler transitions N; this
script separates that from what the implementation actually executes.

    python scripts/analyze_sweep_padding.py
    python scripts/analyze_sweep_padding.py --batch-size 128 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from exp0.dataset import (  # noqa: E402
    Task3SumDataset,
    build_default_vocab,
    pad_collate_fn,
)
from exp0.generation import generate_protocol_packed_instances  # noqa: E402

SWEEP_N = (0, 1, 2, 4, 8, 16, 32, 36)
IGNORE_INDEX = -100


def measure(n_filler: int, batch_size: int, length: int, dimension: int,
            heads: int, chunk_len: int, seed: int) -> Dict[str, Any]:
    """Measure one optimizer batch at the real mixture and batch size."""
    vocab = build_default_vocab(length=length, dimension=dimension)
    packed = generate_protocol_packed_instances(
        batch_size, length=length, dimension=dimension, rng=random.Random(seed))
    dataset = Task3SumDataset(packed, num_filler=n_filler, vocab=vocab, seed=seed,
                              parallel_ratio=0.5, filler_ratio=0.5)
    items = [dataset[i] for i in range(len(dataset))]
    batch = pad_collate_fn(items)

    tuple_positions = int(batch["input_tuples"].shape[1])
    padded_targets = int(batch["targets"].shape[1])
    padded_backbone_t = tuple_positions + padded_targets
    chunk_padded_t = math.ceil(padded_backbone_t / chunk_len) * chunk_len

    by_format: Dict[str, Dict[str, int]] = {}
    logical_positions = 0
    for item in items:
        fmt = item["format"]
        target_len = int(item["targets"].shape[0])
        backbone_t = tuple_positions + target_len
        entry = by_format.setdefault(
            fmt, {"count": 0, "logical_backbone_t": backbone_t})
        entry["count"] += 1
        logical_positions += backbone_t

    # The CE target is loss_mask, padded with -100; targets is padded with the
    # PAD id because it is fed to the model. Supervision is defined by the mask.
    shifted_mask = batch["loss_mask"][:, 1:]
    supervised_tokens = int((shifted_mask != IGNORE_INDEX).sum())

    executed_positions = batch_size * padded_backbone_t
    logical_transitions = logical_positions * heads
    physical_transitions = batch_size * chunk_padded_t * heads
    head_positions = batch_size * (padded_targets - 1)

    return {
        "n_filler": n_filler,
        "samples_by_format": {k: v["count"] for k, v in sorted(by_format.items())},
        "logical_backbone_t_by_format": {
            k: v["logical_backbone_t"] for k, v in sorted(by_format.items())},
        "batch_padded_backbone_t": padded_backbone_t,
        "chunk_padded_backbone_t": chunk_padded_t,
        "logical_model_positions": logical_positions,
        "executed_model_positions": executed_positions,
        "padding_model_positions": executed_positions - logical_positions,
        "supervised_token_positions": supervised_tokens,
        "head_projected_positions": head_positions,
        "logical_recurrent_transitions": logical_transitions,
        "physical_recurrent_transitions": physical_transitions,
        "useful_position_ratio": logical_positions / executed_positions,
        "supervised_of_projected_ratio": supervised_tokens / head_positions,
        "logical_of_physical_transition_ratio": (
            logical_transitions / physical_transitions),
    }


def render(rows: List[Dict[str, Any]]) -> None:
    print(f"\n  {'N':>3} {'fillerT':>8} {'cotT':>5} {'padT':>5} {'chunkT':>7} "
          f"{'useful':>8} {'supervised':>11} {'logical/phys':>13}")
    for row in rows:
        lengths = row["logical_backbone_t_by_format"]
        filler_t = lengths.get("filler", 0)
        cot_t = lengths.get("parallel_cot", 0)
        print(f"  {row['n_filler']:3d} {filler_t:8d} {cot_t:5d} "
              f"{row['batch_padded_backbone_t']:5d} "
              f"{row['chunk_padded_backbone_t']:7d} "
              f"{row['useful_position_ratio']:7.1%} "
              f"{row['supervised_of_projected_ratio']:10.1%} "
              f"{row['logical_of_physical_transition_ratio']:12.1%}")
    print("\n  useful       = logical model positions / executed model positions")
    print("  supervised   = supervised tokens / positions the 32k head projects")
    print("  logical/phys = logical recurrent transitions / physical (chunked)")
    print("\n  The scientific compute budget is the requested filler transitions N.")
    print("  These ratios describe implementation overhead only.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--heads", type=int, default=12,
                        help="hidden_size // head_dim; 768//64 for the 0B model")
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-values", type=int, nargs="+", default=list(SWEEP_N))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = [measure(n, args.batch_size, args.length, args.dimension,
                    args.heads, args.chunk_len, args.seed)
            for n in args.n_values]
    print(f"batch {args.batch_size}, 50/50 mixture, {args.heads} heads, "
          f"CHUNK_LEN {args.chunk_len}")
    render(rows)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "batch_size": args.batch_size, "heads": args.heads,
            "chunk_len": args.chunk_len, "rows": rows}, indent=2) + "\n",
            encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
