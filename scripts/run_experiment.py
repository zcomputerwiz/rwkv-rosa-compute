#!/usr/bin/env python3
"""Run Experiment 0 single run across multiple seeds with fixed validation set."""

import argparse
import json
import random
from pathlib import Path

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, build_default_vocab
from exp0.evaluate import compile_experiment_report
from exp0.task3sum import generate_instance
from exp0.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 0 single configuration across seeds")
    parser.add_argument("--architecture", type=str, default="llama", choices=["llama", "rwkv"])
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--num_filler", type=int, default=None)
    parser.add_argument("--format_type", type=str, default=None, choices=["parallel_cot", "filler", "immediate", "serial_cot", "neutral"])
    parser.add_argument("--parallel_ratio", type=float, default=0.5)
    parser.add_argument("--filler_ratio", type=float, default=0.5)
    parser.add_argument("--serial_ratio", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--eval_seed", type=int, default=9999, help="Fixed eval seed for validation set")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="results/exp0")
    args = parser.parse_args()

    task_cfg = Task3SumConfig(
        length=args.length,
        dimension=args.dimension,
        num_filler=args.num_filler if args.num_filler is not None else args.length**2,
        num_samples=args.num_samples,
    )

    model_cfg = ModelConfig(
        architecture=args.architecture,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        device="cpu",
    )

    vocab = build_default_vocab(length=args.length, dimension=args.dimension)

    # Fixed validation set generated ONCE from eval_seed (T5)
    val_rng = random.Random(args.eval_seed)
    val_instances = [generate_instance(length=args.length, dimension=args.dimension, rng=val_rng) for _ in range(args.val_samples)]
    val_ds = Task3SumDataset(val_instances, format_type="filler", num_filler=task_cfg.num_filler, vocab=vocab, seed=args.eval_seed)

    num_pos = sum(1 for inst in val_instances if inst.has_3sum)
    majority_baseline = max(num_pos, len(val_instances) - num_pos) / len(val_instances)

    per_seed_results = []
    realized_counts_aggregate = {}

    for seed in args.seeds:
        train_rng = random.Random(seed)
        train_instances = [generate_instance(length=args.length, dimension=args.dimension, rng=train_rng) for _ in range(args.num_samples)]

        train_ds = Task3SumDataset(
            train_instances,
            format_type=args.format_type,
            num_filler=task_cfg.num_filler,
            vocab=vocab,
            seed=seed,
            parallel_ratio=args.parallel_ratio,
            filler_ratio=args.filler_ratio,
            serial_ratio=args.serial_ratio,
        )

        for fmt, cnt in train_ds.realized_counts.items():
            realized_counts_aggregate[fmt] = realized_counts_aggregate.get(fmt, 0) + cnt

        train_cfg = TrainConfig(
            seed=seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            mixture=args.format_type if args.format_type else "50_50_cot_filler",
            parallel_ratio=args.parallel_ratio,
            filler_ratio=args.filler_ratio,
            serial_ratio=args.serial_ratio,
        )

        _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)
        history["seed"] = seed
        per_seed_results.append(history)

    report = compile_experiment_report(
        model_cfg,
        train_cfg,
        task_cfg,
        per_seed_results,
        majority_class_baseline=majority_baseline,
        realized_mixture_counts=realized_counts_aggregate,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt_tag = args.format_type if args.format_type else "mix_50_50"
    report_path = out_dir / f"{args.architecture}_len{args.length}_N{task_cfg.num_filler}_fmt_{fmt_tag}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {report_path}")
    print(f"Mean accuracy: {report['metrics']['mean_accuracy']:.4f} (baseline: {majority_baseline:.4f})")


if __name__ == "__main__":
    main()
