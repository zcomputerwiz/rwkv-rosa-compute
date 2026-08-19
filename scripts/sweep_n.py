#!/usr/bin/env python3
"""Run N-filler sweep across N in {0, 1, 2, 4, 8, 16, 32}."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

import scripts.run_experiment as run_experiment
from exp0.rwkv_checkpoint import sha256_file

N_VALUES = [0, 1, 2, 4, 8, 16, 32]


def build_runner_command(
    args: argparse.Namespace,
    n: int,
    run_out: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        "--architecture",
        args.architecture,
        "--rwkv_kernel",
        args.rwkv_kernel,
    ]
    if args.init is not None:
        cmd.extend(["--init", args.init])
    if args.rwkv_checkpoint is not None:
        cmd.extend(["--rwkv_checkpoint", args.rwkv_checkpoint])

    cmd.extend(
        [
            "--hidden_size",
            str(args.hidden_size),
            "--num_hidden_layers",
            str(args.num_hidden_layers),
            "--num_attention_heads",
            str(args.num_attention_heads),
            "--intermediate_size",
            str(args.intermediate_size),
            "--head_dim",
            str(args.head_dim),
            "--num_filler",
            str(n),
            "--length",
            str(args.length),
            "--dimension",
            str(args.dimension),
            "--num_samples",
            str(args.num_samples),
            "--val_samples",
            str(args.val_samples),
            "--eval_seed",
            str(args.eval_seed),
            "--batch_size",
            str(args.batch_size),
            "--learning_rate",
            str(args.learning_rate),
            "--epochs",
            str(args.epochs),
            "--num_workers",
            str(args.num_workers),
            "--val_num_workers",
            str(args.val_num_workers),
            "--prefetch_factor",
            str(args.prefetch_factor),
            "--precision",
            args.precision,
            "--parallel_ratio",
            str(args.parallel_ratio),
            "--filler_ratio",
            str(args.filler_ratio),
            "--serial_ratio",
            str(args.serial_ratio),
            "--device",
            args.device,
            "--out_dir",
            str(run_out),
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
    )
    if args.format_type is not None:
        cmd.extend(["--format_type", args.format_type])
    cmd.append(
        "--vocab_reduction" if args.vocab_reduction else "--no-vocab_reduction"
    )
    cmd.append("--pin_memory" if args.pin_memory else "--no-pin_memory")
    cmd.append("--fused_adamw" if args.fused_adamw else "--no-fused_adamw")
    return cmd


def canonical_sweep_config(args: argparse.Namespace) -> dict:
    """Return a deterministic sweep identity excluding machine-specific paths."""
    config = vars(args).copy()
    config.pop("out_dir", None)
    checkpoint = config.pop("rwkv_checkpoint", None)
    config["rwkv_checkpoint_sha256"] = (
        sha256_file(checkpoint) if checkpoint is not None else None
    )
    config["seeds"] = sorted(config["seeds"])
    config["n_values"] = N_VALUES
    return config


def compute_sweep_id(args: argparse.Namespace) -> str:
    config_json = json.dumps(
        canonical_sweep_config(args),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep N filler token budget for 0A / 0B"
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
    )
    parser.add_argument("--rwkv_checkpoint", type=str, default=None)
    parser.add_argument(
        "--rwkv_kernel",
        type=str,
        default="reference",
        choices=["reference", "cuda"],
    )
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--val_samples", type=int, default=5000)
    parser.add_argument("--eval_seed", type=int, default=9999)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
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
    parser.add_argument(
        "--vocab_reduction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
    )
    parser.add_argument(
        "--fused_adamw",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--out_dir", type=str, default="results/sweeps")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main():
    args = get_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_id = compute_sweep_id(args)
    sweep_root = out_dir / f"{args.architecture}_{sweep_id}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    results = []

    for n in N_VALUES:
        run_out = sweep_root / f"n{n}"
        cmd = build_runner_command(args, n, run_out)

        print(f"=== Running {args.architecture} with N={n} ===")
        subprocess.run(cmd, check=True)

        runner_parser = run_experiment.get_parser()
        parsed_args = runner_parser.parse_args(cmd[2:])
        task_cfg, model_cfg, train_cfg = run_experiment.build_configs(parsed_args)
        summary_path = run_experiment.get_report_path(
            parsed_args,
            task_cfg,
            model_cfg,
            train_cfg,
        )

        if not summary_path.exists():
            raise FileNotFoundError(
                f"Runner completed for N={n}, but expected report was not found: "
                f"{summary_path}"
            )

        with open(summary_path, encoding="utf-8") as f:
            report = json.load(f)

        metrics = report.get("metrics", {})
        results.append(
            {
                "n": n,
                "report_path": str(summary_path),
                "run_id": report.get("run_id"),
                "filler_accuracy": metrics.get("filler_accuracy"),
                "training_answer_accuracy": metrics.get(
                    "best_training_answer_accuracy"
                ),
                "cot_answer_given_cot_accuracy": metrics.get(
                    "cot_answer_given_cot_accuracy"
                ),
                "cot_result_semantic_accuracy": metrics.get(
                    "cot_result_semantic_accuracy"
                ),
                "cot_match_index_accuracy": metrics.get(
                    "cot_match_index_accuracy"
                ),
                "cot_sum_semantic_accuracy": metrics.get(
                    "cot_sum_semantic_accuracy"
                ),
                "cot_result_nll": metrics.get("cot_result_nll"),
            }
        )

    if [result["n"] for result in results] != N_VALUES:
        raise AssertionError("Sweep results are incomplete or out of order.")

    sweep_report = sweep_root / "sweep.json"
    with open(sweep_report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sweep_id": sweep_id,
                "configuration": canonical_sweep_config(args),
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Sweep finished. Summary saved to {sweep_report}")


if __name__ == "__main__":
    main()
