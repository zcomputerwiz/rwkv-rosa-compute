#!/usr/bin/env python3
"""Generate 3SUM datasets and metadata JSON files."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

from exp0.sequences import (
    format_a_parallel_cot,
    format_b_filler,
    format_c_immediate,
    format_d_serial_cot,
    format_e_neutral,
)
from exp0.task3sum import Instance3Sum, generate_instance


def generate_dataset_metadata(
    instances: list[Instance3Sum],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Compute dataset metadata including class balance and majority-class baseline."""
    num_positive = sum(1 for inst in instances if inst.has_3sum)
    num_negative = len(instances) - num_positive
    true_rate = num_positive / len(instances) if instances else 0.0
    majority_class_baseline = max(num_positive, num_negative) / len(instances) if instances else 0.0

    return {
        "length": args.length,
        "dimension": args.dimension,
        "mod": args.mod,
        "num_filler": args.num_filler if args.num_filler is not None else args.length**2,
        "num_samples": len(instances),
        "seed": args.seed,
        "vocab_reduction": args.vocab_reduction,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "true_rate": true_rate,
        "majority_class_baseline": majority_class_baseline,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate 3SUM dataset and metadata")
    parser.add_argument("--length", type=int, default=12, help="Number of input tuples n")
    parser.add_argument("--dimension", type=int, default=3, help="Digits per tuple d")
    parser.add_argument("--mod", type=int, default=10, help="Modulus (default 10)")
    parser.add_argument("--num_filler", type=int, default=None, help="Number of filler tokens N (default n^2)")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of dataset samples")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--vocab_reduction", action="store_true", default=True, help="Enable vocab reduction in CoT")
    parser.add_argument("--out_dir", type=str, default="data/exp0", help="Output directory")
    args = parser.parse_args()

    num_filler = args.num_filler if args.num_filler is not None else args.length**2
    out_dir = Path(args.out_dir) / f"len{args.length}_dim{args.dimension}_N{num_filler}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    instances = []
    for _ in range(args.num_samples):
        inst = generate_instance(
            length=args.length,
            dimension=args.dimension,
            mod=args.mod,
            rng=rng,
        )
        instances.append(inst)

    metadata = generate_dataset_metadata(instances, args)

    # Save metadata JSON
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save raw instance strings
    samples = []
    for inst in instances:
        samples.append({
            "tuples": inst.tuples,
            "has_3sum": inst.has_3sum,
            "matching_indices": inst.matching_indices,
            "parallel_cot": format_a_parallel_cot(inst, vocab_reduction=args.vocab_reduction, rng=rng),
            "filler": format_b_filler(inst, num_filler=num_filler),
            "immediate": format_c_immediate(inst),
            "serial_cot": format_d_serial_cot(inst),
            "neutral": format_e_neutral(inst, num_filler=num_filler),
        })

    data_path = out_dir / "dataset.json"
    with open(data_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Dataset generated at {out_dir}")
    print(f"Samples: {len(instances)}, Majority baseline: {metadata['majority_class_baseline']:.4f}")


if __name__ == "__main__":
    main()
