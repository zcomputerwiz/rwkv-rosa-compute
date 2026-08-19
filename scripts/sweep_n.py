#!/usr/bin/env python3
"""Run N-filler sweep across N in {0, 1, 2, 4, 8, 16, 32}."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


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
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--num_train_samples", type=int, default=50000)
    parser.add_argument("--num_val_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="results/sweeps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    n_values = [0, 1, 2, 4, 8, 16, 32]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for n in n_values:
        run_out = out_dir / f"{args.architecture}_n{n}"
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--architecture",
            args.architecture,
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
            "--num_filler_tokens",
            str(n),
            "--length",
            str(args.length),
            "--dimension",
            str(args.dimension),
            "--vocab_size",
            str(args.vocab_size),
            "--num_train_samples",
            str(args.num_train_samples),
            "--num_val_samples",
            str(args.num_val_samples),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--out_dir",
            str(run_out),
        ]

        print(f"=== Running {args.architecture} with N={n} ===")
        subprocess.run(cmd, check=True)

        summary_path = run_out / "report.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
                results.append({"n": n, "summary": rep.get("aggregate_summary", {})})

    sweep_report = out_dir / f"sweep_{args.architecture}.json"
    with open(sweep_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Sweep finished. Summary saved to {sweep_report}")


if __name__ == "__main__":
    main()
