#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from exp0.config import ModelConfig, TrainConfig
from exp0.train import create_model
from exp1.dataset import PointerChaseDataset
from exp1.pointer_chase import ChaseSpec, generate_dataset, make_neutral_vector
from exp1.train import train_model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train Experiment 1 Pointer Chase")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--num-maps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--precision", type=str, required=True, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--num-silent", type=int, default=0)
    parser.add_argument("--silent-kind", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    spec = ChaseSpec(
        num_nodes=args.num_nodes,
        num_maps=args.num_maps,
        max_depth=max(32, args.depth) # Give it enough max depth capacity
    )

    neutral_vector = make_neutral_vector(spec) if args.silent_kind == "neutral" else None

    print("Generating train dataset...")
    train_inst = generate_dataset(
        args.train_size,
        queries_per_memory=4,
        depth=args.depth,
        seed=args.seed,
        num_nodes=args.num_nodes,
        num_maps=args.num_maps,
    )
    train_dataset = PointerChaseDataset(
        train_inst,
        spec,
        num_silent=args.num_silent,
        silent_kind=args.silent_kind,
        neutral_vector=neutral_vector,
    )

    print("Generating val dataset...")
    val_inst = generate_dataset(
        args.val_size,
        queries_per_memory=4,
        depth=args.depth,
        seed=args.seed + 1,
        num_nodes=args.num_nodes,
        num_maps=args.num_maps,
    )
    val_dataset = PointerChaseDataset(
        val_inst,
        spec,
        num_silent=args.num_silent,
        silent_kind=args.silent_kind,
        neutral_vector=neutral_vector,
    )

    model_cfg = ModelConfig(
        architecture="rwkv",
        hidden_size=args.d_model,
        num_hidden_layers=args.layers,
        num_attention_heads=args.d_model // 64, # match standard head dim
        head_dim=64,
        vocab_size=args.num_nodes,
        rwkv_kernel="reference" if device.type == "cpu" else "cuda",
    )

    model = create_model(
        model_cfg,
        d_input=spec.d_input,
        compact_reduced_features=False
    ).to(device)

    train_cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        precision=args.precision,
        epochs=args.epochs,
    )

    print("Training...")
    model, history = train_model(
        model,
        train_dataset,
        val_dataset,
        spec,
        model_cfg,
        train_cfg,
        device,
    )

    print(f"Done! Best Val Acc: {history['best_val_accuracy']:.4f}")

    # Simple report
    import json
    report_file = args.out_dir / f"report_seed{args.seed}.json"
    with open(report_file, "w") as f:
        json.dump(history, f, indent=2)

    return 0

if __name__ == "__main__":
    sys.exit(main())
