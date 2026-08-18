#!/usr/bin/env python3
"""Run Experiment 0 single run across multiple seeds."""

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
    parser.add_argument("--format_type", type=str, default="filler", choices=["parallel_cot", "filler", "immediate", "serial_cot", "neutral"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=200)
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
        hidden_size=128,          # Scaled for fast iteration in experiment runner
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        device="cpu",
    )

    vocab = build_default_vocab(length=args.length, dimension=args.dimension)

    per_seed_results = []
    majority_baselines = []

    for seed in args.seeds:
        rng = random.Random(seed)

        # Generate train and validation data
        train_instances = [generate_instance(length=args.length, dimension=args.dimension, rng=rng) for _ in range(args.num_samples)]
        val_instances = [generate_instance(length=args.length, dimension=args.dimension, rng=rng) for _ in range(args.val_samples)]

        num_pos = sum(1 for inst in val_instances if inst.has_3sum)
        majority_baselines.append(max(num_pos, len(val_instances) - num_pos) / len(val_instances))

        train_ds = Task3SumDataset(train_instances, format_type=args.format_type, num_filler=task_cfg.num_filler, vocab=vocab, seed=seed)
        val_ds = Task3SumDataset(val_instances, format_type=args.format_type, num_filler=task_cfg.num_filler, vocab=vocab, seed=seed)

        train_cfg = TrainConfig(
            seed=seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            mixture=args.format_type,
        )

        _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)
        history["seed"] = seed
        per_seed_results.append(history)

    avg_majority_baseline = sum(majority_baselines) / len(majority_baselines)
    report = compile_experiment_report(model_cfg, train_cfg, task_cfg, per_seed_results, avg_majority_baseline)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{args.architecture}_len{args.length}_N{task_cfg.num_filler}_fmt_{args.format_type}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {report_path}")
    print(f"Mean accuracy: {report['metrics']['mean_accuracy']:.4f} (baseline: {avg_majority_baseline:.4f})")


if __name__ == "__main__":
    main()
