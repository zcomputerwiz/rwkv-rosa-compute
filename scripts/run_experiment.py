#!/usr/bin/env python3
"""Run Experiment 0 single configuration across seeds."""

import argparse
import json
import random
from pathlib import Path

import torch

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, build_default_vocab
from exp0.evaluate import (
    canonical_run_config,
    compile_experiment_report,
    compute_run_id,
)
from exp0.rwkv_checkpoint import sha256_file
from exp0.task3sum import generate_instance
from exp0.train import train_model


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Experiment 0 single configuration across seeds"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="llama",
        choices=["llama", "rwkv"],
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        choices=["random", "pretrained"],
        help=(
            "Initialization mode. Llama defaults to random. RWKV requires an "
            "explicit checkpoint for pretrained 0B, or --init random for a "
            "debug-only random run."
        ),
    )
    parser.add_argument(
        "--rwkv_checkpoint",
        type=str,
        default=None,
        help="Explicit path to a stock pretrained RWKV-7 x070 checkpoint",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=384,
        help="Model hidden size (default 384)",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=4,
        help="Number of layers (default 4)",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=6,
        help="Transformer attention heads (default 6)",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=1536,
        help="FFN intermediate size (default 1536)",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=64,
        help="RWKV head dimension (default 64)",
    )
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--num_filler", type=int, default=None)
    parser.add_argument(
        "--format_type",
        type=str,
        default=None,
        choices=[
            "parallel_cot",
            "filler",
            "immediate",
            "serial_cot",
            "neutral",
        ],
    )
    parser.add_argument("--parallel_ratio", type=float, default=0.5)
    parser.add_argument("--filler_ratio", type=float, default=0.5)
    parser.add_argument("--serial_ratio", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=2000)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=9999,
        help="Fixed eval seed for validation set",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="results/exp0")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--vocab_reduction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable vocab reduction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of DataLoader workers. On Windows, >0 uses spawn and "
            "pickles dataset copies; start conservatively."
        ),
    )
    return parser


def _resolve_initialization(args: argparse.Namespace) -> tuple[str, str | None, str | None]:
    requested_init = getattr(args, "init", None)
    checkpoint_arg = getattr(args, "rwkv_checkpoint", None)

    if args.architecture == "llama":
        if requested_init not in {None, "random"}:
            raise ValueError(
                "Experiment 0 Llama runs support only random initialization."
            )
        if checkpoint_arg is not None:
            raise ValueError(
                "--rwkv_checkpoint is only valid with --architecture rwkv."
            )
        return "random", None, None

    if args.hidden_size % args.head_dim != 0:
        raise ValueError(
            f"RWKV hidden_size={args.hidden_size} must be divisible by "
            f"head_dim={args.head_dim}."
        )

    if requested_init is None:
        if checkpoint_arg is None:
            raise ValueError(
                "RWKV Experiment 0B requires a stock pretrained checkpoint. "
                "Provide --rwkv_checkpoint PATH (pretrained is inferred), or "
                "use --init random explicitly for a debug-only random run."
            )
        requested_init = "pretrained"

    if requested_init == "random":
        if checkpoint_arg is not None:
            raise ValueError(
                "--init random must not be combined with --rwkv_checkpoint."
            )
        return "random", None, None

    if checkpoint_arg is None:
        raise ValueError(
            "--init pretrained requires --rwkv_checkpoint PATH for RWKV."
        )

    checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RWKV-7 checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    return "pretrained", str(checkpoint_path), checkpoint_sha256


def build_configs(
    args: argparse.Namespace,
) -> tuple[Task3SumConfig, ModelConfig, TrainConfig]:
    init_mode, checkpoint_path, checkpoint_sha256 = _resolve_initialization(args)

    task_cfg = Task3SumConfig(
        length=args.length,
        dimension=args.dimension,
        num_filler=(
            args.num_filler if args.num_filler is not None else args.length**2
        ),
        num_samples=args.num_samples,
        vocab_reduction=args.vocab_reduction,
    )

    num_heads = args.num_attention_heads
    if args.architecture == "rwkv":
        num_heads = args.hidden_size // args.head_dim

    model_cfg = ModelConfig(
        architecture=args.architecture,
        init_mode=init_mode,
        rwkv_checkpoint=checkpoint_path,
        rwkv_checkpoint_sha256=checkpoint_sha256,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=num_heads,
        intermediate_size=args.intermediate_size,
        head_dim=args.head_dim,
        device=args.device,
    )

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        mixture=(
            args.format_type if args.format_type else "50_50_cot_filler"
        ),
        parallel_ratio=args.parallel_ratio,
        filler_ratio=args.filler_ratio,
        serial_ratio=args.serial_ratio,
        num_workers=args.num_workers,
    )
    return task_cfg, model_cfg, train_cfg


def get_report_path(
    args: argparse.Namespace,
    task_cfg: Task3SumConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> Path:
    run_id = compute_run_id(
        model_cfg,
        train_cfg,
        task_cfg,
        args.eval_seed,
        args.val_samples,
        args.seeds,
    )
    fmt_tag = args.format_type if args.format_type else "mix_50_50"
    filename = (
        f"{args.architecture}_len{args.length}_N{task_cfg.num_filler}_"
        f"fmt_{fmt_tag}_{run_id}.json"
    )
    return Path(args.out_dir) / filename


def _check_existing_report(
    report_path: Path,
    current_run_config: dict,
) -> bool:
    """Return True when an existing report exactly matches the requested run."""
    if not report_path.exists():
        return False

    with open(report_path, encoding="utf-8") as f:
        existing_report = json.load(f)

    existing_config = existing_report.get("run_config")
    if existing_config == current_run_config:
        print(
            f"Report {report_path} already exists and its full run_config "
            "matches. Skipping run."
        )
        return True

    raise ValueError(
        f"Report {report_path} exists but its full run_config does not match "
        "the requested configuration. Will not overwrite."
    )


def main():
    parser = get_parser()
    args = parser.parse_args()

    task_cfg, model_cfg, train_cfg_base = build_configs(args)
    if args.val_samples <= 0:
        raise ValueError("val_samples must be greater than zero.")
    if args.num_samples <= 0:
        raise ValueError("num_samples must be greater than zero.")
    if not args.seeds:
        raise ValueError("At least one training seed is required.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = get_report_path(args, task_cfg, model_cfg, train_cfg_base)
    current_run_config = canonical_run_config(
        model_cfg,
        train_cfg_base,
        task_cfg,
        args.eval_seed,
        args.val_samples,
        args.seeds,
    )
    if _check_existing_report(report_path, current_run_config):
        return

    vocab = build_default_vocab(length=args.length, dimension=args.dimension)

    val_rng = random.Random(args.eval_seed)
    val_instances = [
        generate_instance(
            length=args.length,
            dimension=args.dimension,
            mod=task_cfg.mod,
            rng=val_rng,
        )
        for _ in range(args.val_samples)
    ]

    filler_val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        num_filler=task_cfg.num_filler,
        vocab=vocab,
        seed=args.eval_seed,
        vocab_reduction=args.vocab_reduction,
    )
    cot_val_ds = Task3SumDataset(
        val_instances,
        format_type="parallel_cot",
        num_filler=task_cfg.num_filler,
        vocab=vocab,
        seed=args.eval_seed,
        vocab_reduction=args.vocab_reduction,
    )

    num_pos = sum(1 for inst in val_instances if inst.has_3sum)
    majority_baseline = max(
        num_pos,
        len(val_instances) - num_pos,
    ) / len(val_instances)

    per_seed_results = []
    realized_counts_aggregate: dict[str, int] = {}

    for seed in args.seeds:
        train_rng = random.Random(seed)
        train_instances = [
            generate_instance(
                length=args.length,
                dimension=args.dimension,
                mod=task_cfg.mod,
                rng=train_rng,
            )
            for _ in range(args.num_samples)
        ]

        train_ds = Task3SumDataset(
            train_instances,
            format_type=args.format_type,
            num_filler=task_cfg.num_filler,
            vocab=vocab,
            seed=seed,
            vocab_reduction=args.vocab_reduction,
            parallel_ratio=args.parallel_ratio,
            filler_ratio=args.filler_ratio,
            serial_ratio=args.serial_ratio,
        )

        for fmt, count in train_ds.realized_counts.items():
            realized_counts_aggregate[fmt] = (
                realized_counts_aggregate.get(fmt, 0) + count
            )

        train_cfg = TrainConfig(
            seed=seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            mixture=(
                args.format_type if args.format_type else "50_50_cot_filler"
            ),
            parallel_ratio=args.parallel_ratio,
            filler_ratio=args.filler_ratio,
            serial_ratio=args.serial_ratio,
            num_workers=args.num_workers,
        )

        _, history = train_model(
            model_cfg,
            train_cfg,
            task_cfg,
            train_ds,
            filler_val_dataset=filler_val_ds,
            cot_val_dataset=cot_val_ds,
        )
        history["seed"] = seed
        history["training_seed"] = seed
        history["task_seed"] = seed
        per_seed_results.append(history)

    report = compile_experiment_report(
        model_cfg,
        train_cfg_base,
        task_cfg,
        per_seed_results,
        majority_class_baseline=majority_baseline,
        realized_mixture_counts=realized_counts_aggregate,
        eval_seed=args.eval_seed,
        val_samples=args.val_samples,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {report_path}")
    print(
        f"Mean accuracy: {report['metrics']['mean_accuracy']:.4f} "
        f"(baseline: {majority_baseline:.4f})"
    )


if __name__ == "__main__":
    main()
