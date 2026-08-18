#!/usr/bin/env python3
"""Run N-filler sweep across N in {0, 1, 2, 4, 8, 16, 32}."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--n_values", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--parallel_ratio", type=float, default=0.5)
    parser.add_argument("--filler_ratio", type=float, default=0.5)
    parser.add_argument("--format_type", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="results/sweeps")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_summary = {}

    for n in args.n_values:
        print(f"\n--- Running sweep point N = {n} ---")
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--architecture", args.architecture,
            "--hidden_size", str(args.hidden_size),
            "--num_hidden_layers", str(args.num_hidden_layers),
            "--num_attention_heads", str(args.num_attention_heads),
            "--intermediate_size", str(args.intermediate_size),
            "--head_dim", str(args.head_dim),
            "--length", str(args.length),
            "--dimension", str(args.dimension),
            "--num_filler", str(n),
            "--parallel_ratio", str(args.parallel_ratio),
            "--filler_ratio", str(args.filler_ratio),
            "--num_samples", str(args.num_samples),
            "--val_samples", str(args.val_samples),
            "--epochs", str(args.epochs),
            "--seeds", *[str(s) for s in args.seeds],
            "--out_dir", str(out_dir),
        ]
        if args.format_type:
            cmd.extend(["--format_type", args.format_type])

        subprocess.run(cmd, check=True)

        fmt_tag = args.format_type if args.format_type else "mix_50_50"
        report_file = out_dir / f"{args.architecture}_len{args.length}_N{n}_fmt_{fmt_tag}.json"
        if report_file.exists():
            with open(report_file) as f:
                rep = json.load(f)
                sweep_summary[f"N_{n}"] = rep["metrics"]

    summary_file = out_dir / f"sweep_{args.architecture}_len{args.length}_dim{args.dimension}.json"
    with open(summary_file, "w") as f:
        json.dump(sweep_summary, f, indent=2)

    print(f"\nSweep complete. Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
