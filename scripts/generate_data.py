#!/usr/bin/env python3
"""Generate 3SUM datasets and metadata JSON files."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

from exp0.generation import generate_protocol_packed_instances
from exp0.sequences import (
    format_a_parallel_cot,
    format_b_filler,
    format_c_immediate,
    format_d_serial_cot,
    format_e_neutral,
)
from exp0.task3sum import (
    DEFAULT_CORRUPTION_RATE,
    GENERATOR_MODES,
    SOURCE_GENERATOR,
    Instance3Sum,
)


def generate_dataset_metadata(
    instances: list[Instance3Sum],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Compute dataset metadata including realized class balance."""
    num_positive = sum(1 for inst in instances if inst.has_3sum)
    num_negative = len(instances) - num_positive
    realized_positive_rate = num_positive / len(instances) if instances else 0.0
    majority_class_baseline = (
        max(num_positive, num_negative) / len(instances) if instances else 0.0
    )

    return {
        "length": args.length,
        "dimension": args.dimension,
        "mod": args.mod,
        "num_filler": (
            args.num_filler if args.num_filler is not None else args.length**2
        ),
        "num_samples": len(instances),
        "seed": args.seed,
        "vocab_reduction": args.vocab_reduction,
        "generator_mode": args.generator_mode,
        "requested_true_construction_rate": args.true_rate,
        "corruption_rate": args.corruption_rate,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "true_rate": realized_positive_rate,
        "realized_positive_rate": realized_positive_rate,
        "majority_class_baseline": majority_class_baseline,
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate 3SUM dataset and metadata"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=12,
        help="Number of input tuples n",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=3,
        help="Digits per tuple d",
    )
    parser.add_argument(
        "--mod",
        type=int,
        default=10,
        help="Modulus (must be 10 for Experiment 0)",
    )
    parser.add_argument(
        "--num_filler",
        type=int,
        default=None,
        help="Number of filler tokens N (default n^2)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of dataset samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed",
    )
    parser.add_argument(
        "--true_rate",
        type=float,
        default=0.5,
        help=(
            "Probability of selecting the planted-positive construction arm. "
            "In source_corrupted mode the realized positive-label rate can be higher."
        ),
    )
    parser.add_argument(
        "--generator_mode",
        type=str,
        default=SOURCE_GENERATOR,
        choices=list(GENERATOR_MODES),
    )
    parser.add_argument(
        "--corruption_rate",
        type=float,
        default=DEFAULT_CORRUPTION_RATE,
    )
    parser.add_argument(
        "--vocab_reduction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable vocab reduction in CoT",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/exp0",
        help="Output directory",
    )
    return parser


def get_dataset_output_dir(args: argparse.Namespace) -> Path:
    """Return a collision-resistant path for generated dataset artifacts."""
    num_filler = (
        args.num_filler if args.num_filler is not None else args.length**2
    )
    vocab_tag = "vred" if args.vocab_reduction else "fullvocab"
    generator_mode = getattr(args, "generator_mode", SOURCE_GENERATOR)
    true_rate = getattr(args, "true_rate", 0.5)
    corruption_rate = getattr(args, "corruption_rate", DEFAULT_CORRUPTION_RATE)
    run_name = (
        f"len{args.length}_dim{args.dimension}_N{num_filler}_"
        f"S{args.num_samples}_seed{args.seed}_{vocab_tag}_"
        f"gen-{generator_mode}_tr{true_rate:g}_cr{corruption_rate:g}"
    )
    return Path(args.out_dir) / run_name


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.mod != 10:
        parser.error(
            f"Experiment 0 supports only --mod 10; received --mod {args.mod}."
        )
    if not 0.0 <= args.true_rate <= 1.0:
        parser.error("--true_rate must be in [0, 1].")
    if args.corruption_rate < 1.0:
        parser.error("--corruption_rate must be >= 1.0.")

    num_filler = (
        args.num_filler if args.num_filler is not None else args.length**2
    )
    out_dir = get_dataset_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    packed = generate_protocol_packed_instances(
        args.num_samples,
        length=args.length,
        dimension=args.dimension,
        mod=args.mod,
        true_rate=args.true_rate,
        rng=random.Random(args.seed),
        generator_mode=args.generator_mode,
        corruption_rate=args.corruption_rate,
    )
    instances = [packed.instance_at(idx) for idx in range(len(packed))]
    metadata = generate_dataset_metadata(instances, args)

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    samples = []
    for idx, inst in enumerate(instances):
        format_rng = random.Random(f"{args.seed}_{idx}")
        samples.append(
            {
                "tuples": inst.tuples,
                "has_3sum": inst.has_3sum,
                "matching_indices": inst.matching_indices,
                "parallel_cot": format_a_parallel_cot(
                    inst,
                    vocab_reduction=args.vocab_reduction,
                    rng=format_rng,
                ),
                "filler": format_b_filler(inst, num_filler=num_filler),
                "immediate": format_c_immediate(inst),
                "serial_cot": format_d_serial_cot(inst),
                "neutral": format_e_neutral(inst, num_filler=num_filler),
            }
        )

    data_path = out_dir / "dataset.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"Dataset generated at {out_dir}")
    print(
        f"Samples: {len(instances)}, "
        f"Majority baseline: {metadata['majority_class_baseline']:.4f}"
    )


if __name__ == "__main__":
    main()
