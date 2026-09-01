#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

# Add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from exp0.config import ModelConfig, TrainConfig
from exp0.train import create_model, set_seed
from exp1.dataset import PointerChaseDataset
from exp1.pointer_chase import ChaseSpec, generate_dataset, make_neutral_vector
from exp1.qwen4_micro import (
    QWEN4_MAX_POSITION_EMBEDDINGS,
    QWEN4_VARIANTS,
    Qwen4MicroConfig,
    create_qwen4_micro_model,
)
from exp1.train import _trainable, evaluate_vway_accuracy, train_model
from exp1.workspace import Workspace
from rosa_compute.diagnostics import get_artifact_environment


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
    parser.add_argument(
        "--architecture",
        choices=["rwkv", "qwen4_exp"],
        default="rwkv",
    )
    parser.add_argument(
        "--qwen4-variant",
        choices=QWEN4_VARIANTS,
        default=None,
        help="defaults to hybrid; valid only with --architecture qwen4_exp",
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--num-silent", type=int, default=0)
    parser.add_argument("--silent-kind", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    # The 2x2 screen. M=1,K=1 is the baseline CELL, not a no-workspace run:
    # it still routes and refines, and is parameter-matched to every other cell.
    parser.add_argument("--num-slots", type=int, default=None,
                        help="M: workspace slots; defaults to 1 with --workspace")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="K: refinement steps; defaults to 1 with --workspace")
    parser.add_argument("--m-max", type=int, default=None,
                        help="offset table size; defaults to 8 with --workspace")
    parser.add_argument("--workspace", action="store_true",
                        help="enable the latent workspace; Gate 0 leaves this off")
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

    # The reference recurrence is a Python loop over timesteps, so a step is
    # dominated by kernel-launch overhead rather than by arithmetic: measured
    # 848 instances/s eager against 2947 compiled and 8177 under cudagraphs, at
    # a batch size the GPU could serve four times over. Opt-in, because a gate
    # should not silently change its execution path.
    #
    # Scope of the equivalence claim, stated precisely because it is easy to
    # overstate: only `model.backbone` is compiled. The head, the loss, gradient
    # clipping, and the optimizer step all remain eager, so this is not a
    # captured training step. What was measured is that the per-epoch mean loss
    # and validation accuracy matched eager exactly for 15 epochs at D=1, N=0,
    # batch 64, fp32. Parameters, gradients and per-step losses were not
    # compared, and `set_seed` does not enable deterministic algorithms -- so
    # this is strong evidence of an unperturbed trajectory at this scale, not a
    # proof of bitwise equivalence in general.
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the backbone; see --help note on batch divisibility")
    # Default resolved after parsing rather than here, so that passing a backend
    # without --compile is a visible error instead of a silently ignored flag.
    parser.add_argument("--compile-backend", type=str, default=None,
                        choices=["inductor", "cudagraphs"],
                        help="default cudagraphs: it captures the backbone's "
                             "forward as a graph, and reproduced eager's "
                             "per-epoch loss and accuracy exactly over 15 "
                             "epochs where inductor drifted ~1.2e-3. Requires "
                             "every batch to be the same shape")

    # Deliberate memorization: evaluate on the training set itself. The bank
    # from --val-data-seed is still generated and scored once after training,
    # as a non-gating held-out diagnostic.
    parser.add_argument("--overfit-train-as-val", action="store_true",
                        help="evaluate on the training set; held-out bank becomes a diagnostic")

    args = parser.parse_args(argv)
    if args.compile_backend is not None and not args.compile:
        parser.error("--compile-backend requires --compile")
    if args.architecture != "qwen4_exp" and args.qwen4_variant is not None:
        parser.error("--qwen4-variant requires --architecture qwen4_exp")
    if args.architecture == "qwen4_exp" and args.rwkv_kernel is not None:
        parser.error("--rwkv-kernel is valid only with --architecture rwkv")
    if args.architecture == "qwen4_exp" and args.compile:
        parser.error(
            "Qwen4-Exp compilation is not part of the registered pilot; use eager execution"
        )
    if args.architecture == "qwen4_exp" and (
        args.d_model != 128 or args.layers != 4
    ):
        parser.error(
            "The registered Qwen4-Exp pilot requires --d-model 128 --layers 4"
        )
    if args.architecture == "qwen4_exp" and args.precision != "fp32":
        parser.error(
            "The registered Qwen4-Exp pilot requires --precision fp32"
        )
    if not args.workspace and any(
        value is not None for value in (args.num_slots, args.num_steps, args.m_max)
    ):
        parser.error("--num-slots, --num-steps, and --m-max require --workspace")
    num_slots = 1 if args.num_slots is None else args.num_slots
    num_steps = 1 if args.num_steps is None else args.num_steps
    m_max = 8 if args.m_max is None else args.m_max
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_seed = args.seed if args.model_seed is None else args.model_seed
    train_data_seed = args.seed if args.train_data_seed is None else args.train_data_seed
    val_data_seed = args.seed + 1 if args.val_data_seed is None else args.val_data_seed

    device = torch.device(args.device)
    rwkv_kernel = None
    if args.architecture == "rwkv":
        rwkv_kernel = args.rwkv_kernel or (
            "reference" if device.type == "cpu" else "cuda"
        )

    spec = ChaseSpec(
        num_nodes=args.num_nodes,
        num_maps=args.num_maps,
        max_depth=max(32, args.depth) # Give it enough max depth capacity
    )
    if (
        args.architecture == "qwen4_exp"
        and spec.seq_len(args.num_silent) > QWEN4_MAX_POSITION_EMBEDDINGS
    ):
        parser.error(
            "Qwen4-Exp input length exceeds the registered "
            f"{QWEN4_MAX_POSITION_EMBEDDINGS}-position capacity"
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

    set_seed(model_seed)
    qwen4_variant = None
    if args.architecture == "rwkv":
        model_cfg = ModelConfig(
            architecture="rwkv",
            hidden_size=args.d_model,
            num_hidden_layers=args.layers,
            num_attention_heads=args.d_model // 64, # match standard head dim
            head_dim=64,
            vocab_size=args.num_nodes,
            rwkv_kernel=rwkv_kernel,
        )
        model = create_model(
            model_cfg,
            d_input=spec.d_input,
            compact_reduced_features=False
        ).to(device)
    else:
        qwen4_variant = args.qwen4_variant or "hybrid"
        model_cfg = Qwen4MicroConfig(
            vocab_size=args.num_nodes,
            hidden_size=args.d_model,
            num_hidden_layers=args.layers,
            variant=qwen4_variant,
        )
        model = create_qwen4_micro_model(
            model_cfg,
            d_input=spec.d_input,
        ).to(device)

    compile_backend = (args.compile_backend or "cudagraphs") if args.compile else None

    if args.compile:
        # A ragged final batch is another input shape. Under inductor that costs
        # a second compile -- measured at 161 s against 11 s for the first, more
        # than compiling saves on a short run. Under cudagraphs it is fatal: the
        # replay copies into fixed-size buffers and raises
        #   RuntimeError: The size of tensor a (64) must match ... (16)
        # partway through the first evaluation. So it is refused rather than
        # warned about for that backend, before any training is done.
        #
        # Both banks matter. Checking only the training bank is what made a
        # first attempt here measure 1.02x instead of 3.1x, and the gate's own
        # ratified 5000/500 is ragged in both.
        #
        # Sizes count memories while the batch counts instances, so the aligned
        # memory counts are the multiples of
        # batch / gcd(batch, queries_per_memory), not the instance count rounded
        # down. At memories=5, q=4, batch=6 the naive rounding suggests 4, whose
        # 16 instances are still ragged modulo 6; the answer is 3.
        step = args.batch_size // math.gcd(args.batch_size, args.queries_per_memory)
        ragged = []
        for name, memories in (("train", args.train_size), ("val", args.val_size)):
            instances = memories * args.queries_per_memory
            remainder = instances % args.batch_size
            if not remainder:
                continue
            lower = memories - memories % step
            fix = (f"--{name}-size {lower}" if lower
                   else f"--{name}-size {step} or larger")
            ragged.append(f"{instances} {name} instances is not a multiple of "
                          f"batch {args.batch_size} (remainder {remainder}); "
                          f"aligned sizes are multiples of {step}, so use {fix}")
        if ragged:
            detail = "; ".join(ragged)
            if compile_backend == "cudagraphs":
                parser.error(
                    f"--compile-backend cudagraphs requires every batch to have "
                    f"the same shape, and {detail}. Align the banks, or pass "
                    f"--compile-backend inductor, which tolerates a ragged batch "
                    f"at the cost of an extra compile."
                )
            print(f"warning: {detail}. The ragged final batch forces an extra "
                  f"compile that can cost more than compiling saves.")

        # Compiled on the backbone rather than on forward_logits, because
        # training and evaluation both reach the model through it. Compiling one
        # call site would run a compiled architecture and score an eager one --
        # the same train/eval split forward_logits exists to prevent.
        model.backbone = torch.compile(model.backbone, backend=compile_backend)

    train_cfg = TrainConfig(
        seed=model_seed,
        batch_size=args.batch_size,
        precision=args.precision,
        epochs=args.epochs,
    )

    workspace = None
    if args.workspace:
        workspace = Workspace(
            model_cfg.hidden_size,
            num_slots=num_slots,
            num_steps=num_steps,
            m_max=m_max,
        )
        learned = sum(p.numel() for p in workspace.parameters() if p.requires_grad)
        print(f"workspace: M={num_slots} K={num_steps} "
              f"learned={learned:,} (invariant across the 2x2)")

    print("Training...")
    model, history = train_model(
        model,
        train_dataset,
        eval_dataset,
        spec,
        model_cfg,
        train_cfg,
        device,
        workspace=workspace,
        checkpoint_path=args.checkpoint_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        depth=args.depth,
        num_silent=args.num_silent,
        silent_kind=args.silent_kind,
        queries_per_memory=args.queries_per_memory,
        train_data_seed=train_data_seed,
        val_data_seed=val_data_seed,
        train_size=args.train_size,
        val_size=args.val_size,
        overfit_train_as_val=args.overfit_train_as_val,
        compile_backend=compile_backend,
    )

    # The gates are fixed-budget with no checkpoint selection, so the outcome is
    # the final epoch. best_val_accuracy is a maximum over epochs and would be
    # selection by another name.
    epoch_accuracies = history["epoch_val_accuracies"]
    final_accuracy = epoch_accuracies[-1] if epoch_accuracies else None

    holdout_diagnostic = None
    if args.overfit_train_as_val:
        holdout_diagnostic = evaluate_vway_accuracy(
            model, holdout_dataset, train_cfg, device, workspace
        )

    print(f"Done! Final {eval_target} acc: {final_accuracy:.4f}"
          if final_accuracy is not None else "Done! No epochs ran.")
    if holdout_diagnostic is not None:
        print(f"  held-out diagnostic (non-gating): {holdout_diagnostic:.4f}")

    # Recorded in the report itself, not only in the run_with_provenance
    # wrapper, so a report is self-describing when read on its own. It matters
    # here specifically because --compile changes the execution path, and a
    # claim about a compiled run is not interpretable without knowing which
    # torch, CUDA and GPU produced it.
    environment = get_artifact_environment()

    report = {
        "eval_target": eval_target,
        "final_accuracy": final_accuracy,
        "environment": environment,
        "epoch_accuracies": epoch_accuracies,
        "holdout_diagnostic_accuracy": holdout_diagnostic,
        "history": history,
        "config": {
            "architecture": args.architecture,
            "depth": args.depth,
            "num_nodes": args.num_nodes,
            "num_maps": args.num_maps,
            "d_model": args.d_model,
            "layers": args.layers,
            "precision": args.precision,
            "rwkv_kernel": rwkv_kernel,
            "qwen4_variant": qwen4_variant,
            "qwen4_config": (
                model_cfg.resolved()
                if isinstance(model_cfg, Qwen4MicroConfig)
                else None
            ),
            "trainable_parameters": sum(
                p.numel() for p in _trainable(model, workspace)
                if p.requires_grad
            ),
            "compile": args.compile,
            "compile_backend": compile_backend,
            "device": str(device),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": train_cfg.learning_rate,
            "weight_decay": train_cfg.weight_decay,
            "grad_clip": train_cfg.grad_clip,
            "adam_betas": [train_cfg.adam_beta1, train_cfg.adam_beta2],
            "lr_schedule": train_cfg.lr_schedule,
            "warmup_fraction": train_cfg.warmup_fraction,
            "planned_optimizer_steps": (
                (len(train_dataset) + args.batch_size - 1)
                // args.batch_size
                * args.epochs
            ),
            "checkpoint_path": (
                str(args.checkpoint_path.expanduser().resolve())
                if args.checkpoint_path is not None
                else None
            ),
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
            "workspace": workspace is not None,
            "num_slots": num_slots if workspace is not None else None,
            "num_steps": num_steps if workspace is not None else None,
            "m_max": m_max if workspace is not None else None,
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
