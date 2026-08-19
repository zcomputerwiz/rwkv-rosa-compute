#!/usr/bin/env python3
"""Run N-filler sweep across N in {0, 1, 2, 4, 8, 16, 32}."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

import scripts.run_experiment as run_experiment


def build_runner_command(args, n: int, run_out: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/run_experiment.py",
        "--architecture", args.architecture,
        "--hidden_size", str(args.hidden_size),
        "--num_hidden_layers", str(args.num_hidden_layers),
        "--num_attention_heads", str(args.num_attention_heads),
        "--intermediate_size", str(args.intermediate_size),
        "--head_dim", str(args.head_dim),
        "--num_filler", str(n),
        "--length", str(args.length),
        "--dimension", str(args.dimension),
        "--num_samples", str(args.num_samples),
        "--val_samples", str(args.val_samples),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--epochs", str(args.epochs),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--out_dir", str(run_out),
    ]


def main():
    parser = argparse.ArgumentParser(description="Sweep N filler token budget for 0A / 0B")
    parser.add_argument("--architecture", type=str, default="llama", choices=["llama", "rwkv"])
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--val_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/sweeps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    n_values = [0, 1, 2, 4, 8, 16, 32]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for n in n_values:
        run_out = out_dir / f"{args.architecture}_n{n}"
        cmd = build_runner_command(args, n, run_out)

        print(f"=== Running {args.architecture} with N={n} ===")
        subprocess.run(cmd, check=True)

        # Reconstruct exactly what run_experiment would have computed for report_path
        parser = run_experiment.get_parser()
        cmd_args = cmd[2:] # skip sys.executable and script name
        parsed_args = parser.parse_args(cmd_args)

        task_cfg, model_cfg, train_cfg = run_experiment.build_configs(parsed_args)
        summary_path = run_experiment.get_report_path(parsed_args, task_cfg, model_cfg, train_cfg)

        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                rep = json.load(f)

                metrics = rep.get("metrics", {})
                res = {"n": n}
                if "filler_accuracy" in metrics:
                    res["filler_accuracy"] = metrics["filler_accuracy"]
                if "cot_accuracy" in metrics:
                    res["cot_accuracy"] = metrics["cot_accuracy"]

                results.append(res)

    sweep_report = out_dir / f"sweep_{args.architecture}.json"
    with open(sweep_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Sweep finished. Summary saved to {sweep_report}")


if __name__ == "__main__":
    main()
