#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from exp0.config import ModelConfig, TrainConfig
from exp0.train import create_model, set_seed
from exp1.dataset import PointerChaseDataset
from exp1.pointer_chase import ChaseSpec, generate_dataset, make_neutral_vector
from exp1.train import evaluate_vway_accuracy, train_model


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
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)

    # --seed alone cannot express a gate that fixes the data and varies only the
    # model, because it drives the train bank, the held-out bank, the model
    # initialization, and the training RNG at once. These split it. Each falls
    # back to the --seed-derived value it replaces, so existing invocations are
    # unchanged.
    parser.add_argument("--model-seed", type=int, default=None,
                        help="initialization and training RNG; defaults to --seed")
    parser.add_argument("--train-data-seed", type=int, default=None,
                        help="training bank; defaults to --seed")
    parser.add_argument("--val-data-seed", type=int, default=None,
                        help="held-out bank; defaults to --seed + 1")

    # --train-size and --val-size are counts of *memories*, and each memory
    # yields this many query instances. A budget written in instances is wrong
    # by this factor unless it is stated.
    parser.add_argument("--queries-per-memory", type=int, default=4)

    # Previously implied by the device: reference on CPU, cuda otherwise. A gate
    # that specifies the reference path could not run on a GPU queue.
    parser.add_argument("--rwkv-kernel", type=str, default=None,
                        choices=["reference", "cuda"],
                        help="defaults to reference on CPU, cuda otherwise")

    # Deliberate memorization: evaluate on the training set itself. The bank
    # from --val-data-seed is still generated and scored once after training,
    # as a non-gating held-out diagnostic.
    parser.add_argument("--overfit-train-as-val", action="store_true",
                        help="evaluate on the training set; held-out bank becomes a diagnostic")

    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_seed = args.seed if args.model_seed is None else args.model_seed
    train_data_seed = args.seed if args.train_data_seed is None else args.train_data_seed
    val_data_seed = args.seed + 1 if args.val_data_seed is None else args.val_data_seed

    device = torch.device(args.device)
    rwkv_kernel = args.rwkv_kernel or ("reference" if device.type == "cpu" else "cuda")

    spec = ChaseSpec(
        num_nodes=args.num_nodes,
        num_maps=args.num_maps,
        max_depth=max(32, args.depth) # Give it enough max depth capacity
    )

    neutral_vector = make_neutral_vector(spec) if args.silent_kind == "neutral" else None

    def build(num_memories: int, seed: int) -> PointerChaseDataset:
        instances = generate_dataset(
            num_memories,
            queries_per_memory=args.queries_per_memory,
            depth=args.depth,
            seed=seed,
            num_nodes=args.num_nodes,
            num_maps=args.num_maps,
        )
        return PointerChaseDataset(
            instances,
            spec,
            num_silent=args.num_silent,
            silent_kind=args.silent_kind,
            neutral_vector=neutral_vector,
        )

    print("Generating train dataset...")
    train_dataset = build(args.train_size, train_data_seed)

    print("Generating val dataset...")
    holdout_dataset = build(args.val_size, val_data_seed)

    eval_dataset = train_dataset if args.overfit_train_as_val else holdout_dataset
    eval_target = "train_set" if args.overfit_train_as_val else "held_out"

    model_cfg = ModelConfig(
        architecture="rwkv",
        hidden_size=args.d_model,
        num_hidden_layers=args.layers,
        num_attention_heads=args.d_model // 64, # match standard head dim
        head_dim=64,
        vocab_size=args.num_nodes,
        rwkv_kernel=rwkv_kernel,
    )

    set_seed(model_seed)
    model = create_model(
        model_cfg,
        d_input=spec.d_input,
        compact_reduced_features=False
    ).to(device)

    train_cfg = TrainConfig(
        seed=model_seed,
        batch_size=args.batch_size,
        precision=args.precision,
        epochs=args.epochs,
    )

    print("Training...")
    model, history = train_model(
        model,
        train_dataset,
        eval_dataset,
        spec,
        model_cfg,
        train_cfg,
        device,
        checkpoint_path=args.checkpoint_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # The gates are fixed-budget with no checkpoint selection, so the outcome is
    # the final epoch. best_val_accuracy is a maximum over epochs and would be
    # selection by another name.
    epoch_accuracies = history["epoch_val_accuracies"]
    final_accuracy = epoch_accuracies[-1] if epoch_accuracies else None

    holdout_diagnostic = None
    if args.overfit_train_as_val:
        holdout_diagnostic = evaluate_vway_accuracy(
            model, holdout_dataset, train_cfg, device
        )

    print(f"Done! Final {eval_target} acc: {final_accuracy:.4f}"
          if final_accuracy is not None else "Done! No epochs ran.")
    if holdout_diagnostic is not None:
        print(f"  held-out diagnostic (non-gating): {holdout_diagnostic:.4f}")

    report = {
        "eval_target": eval_target,
        "final_accuracy": final_accuracy,
        "epoch_accuracies": epoch_accuracies,
        "holdout_diagnostic_accuracy": holdout_diagnostic,
        "history": history,
        "config": {
            "depth": args.depth,
            "num_nodes": args.num_nodes,
            "num_maps": args.num_maps,
            "d_model": args.d_model,
            "layers": args.layers,
            "precision": args.precision,
            "rwkv_kernel": rwkv_kernel,
            "device": str(device),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "num_silent": args.num_silent,
            "silent_kind": args.silent_kind,
            "queries_per_memory": args.queries_per_memory,
            "train_memories": args.train_size,
            "train_instances": args.train_size * args.queries_per_memory,
            "val_memories": args.val_size,
            "val_instances": args.val_size * args.queries_per_memory,
            "seed": args.seed,
            "model_seed": model_seed,
            "train_data_seed": train_data_seed,
            "val_data_seed": val_data_seed,
            "overfit_train_as_val": args.overfit_train_as_val,
        },
    }

    # Named for the seed that actually distinguishes the run. The gates vary
    # --model-seed while holding the data banks fixed, so naming this after the
    # legacy --seed would let three gate attempts overwrite one file. Refuse to
    # overwrite either way: a silently clobbered gate result is worse than a
    # failed launch.
    report_file = args.out_dir / f"report_model_seed{model_seed}.json"
    if report_file.exists():
        raise SystemExit(f"refusing to overwrite existing report: {report_file}")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    return 0

if __name__ == "__main__":
    sys.exit(main())
