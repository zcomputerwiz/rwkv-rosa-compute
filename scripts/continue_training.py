#!/usr/bin/env python3
"""Continue a completed Experiment 0 run for additional epochs.

This exists because a completed run cannot simply be resumed for more epochs.
``epochs`` is part of the checkpoint signature, so asking for more is rejected,
and the ``linear_warmup_decay`` schedule has already reached exactly zero -- so
continuing at the stored learning rate would make no parameter updates at all.
Extending therefore requires a *new* schedule, which is an intervention.

The output is deliberately marked as a continuation and is **not** a canonical
Experiment 0 result. It does not carry the source run's ``run_id``, it is not a
fixed-budget run, and it must not be placed on an accuracy-vs-N curve beside
runs that stopped at their planned budget. It answers one question: does the
metric still move if training continues?

Model and optimizer state are both restored, so the optimizer trajectory
genuinely continues and only the schedule is new. Every config is reconstructed
from the checkpoint's own signature rather than re-specified on the command
line, so a continuation cannot silently train on different data than the run it
extends.

    python scripts/continue_training.py <checkpoint.pt> --additional-epochs 1
    python scripts/continue_training.py <checkpoint.pt> --peak-lr 2e-5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig  # noqa: E402
from exp0.dataset import (  # noqa: E402
    Task3SumDataset,
    build_default_vocab,
    pad_collate_fn,
)
from exp0.generation import generate_protocol_packed_instances  # noqa: E402
from exp0.train import create_model, evaluate_accuracy  # noqa: E402

CONTINUATION_VERSION = 1


def configs_from_signature(signature: Dict[str, Any]):
    """Rebuild the exact configs the source run used.

    Reconstructing rather than re-specifying is the safety property of this
    tool: a continuation cannot accidentally use a different model, mixture, or
    data distribution than the run it claims to extend.
    """
    known_train = {f for f in TrainConfig.__dataclass_fields__}
    known_task = {f for f in Task3SumConfig.__dataclass_fields__}
    known_model = {f for f in ModelConfig.__dataclass_fields__}
    train_cfg = TrainConfig(**{k: v for k, v in signature["training"].items()
                               if k in known_train})
    task_cfg = Task3SumConfig(**{k: v for k, v in signature["task"].items()
                                 if k in known_task})
    model_cfg = ModelConfig(**{k: v for k, v in signature["model"].items()
                               if k in known_model})
    return model_cfg, train_cfg, task_cfg


def build_schedule(optimizer, total_steps: int, warmup_fraction: float) -> LambdaLR:
    """Fresh warmup-and-decay over the continuation only."""
    warmup = max(1, int(total_steps * warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step <= warmup:
            return step / warmup
        return max(0.0, 1.0 - (step / total_steps))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def restore_optimizer(optimizer, state: Dict[str, Any], peak_lr: float,
                      device: torch.device) -> None:
    """Load optimizer state, then override the stored (zero) learning rate."""
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        group["lr"] = peak_lr
        group["initial_lr"] = peak_lr
    for buffers in optimizer.state.values():
        for key, value in buffers.items():
            if torch.is_tensor(value):
                buffers[key] = value.to(device)


def supervised_cross_entropy(logits, batch, criterion, device) -> Any:
    """Cross entropy over supervised positions only.

    ``loss_mask`` is the CE target, not ``targets``. ``targets`` is padded with
    the PAD id because it is fed to the model, so using it would count padded
    positions and train the model to predict PAD -- a different objective than
    the run being continued. ``loss_mask`` is padded with -100 and is ignored.
    """
    shift_targets = batch["loss_mask"][:, 1:].reshape(-1).to(device)
    return criterion(logits.reshape(-1, logits.size(-1)), shift_targets)


def last_nonzero_lr(progress: Dict[str, Any]) -> Optional[float]:
    rates = (progress.get("completed") or {}).get("epoch_end_learning_rates") or []
    for rate in reversed(rates):
        if rate:
            return float(rate)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--additional-epochs", type=int, default=1)
    parser.add_argument("--peak-lr", type=float, default=None,
                        help="default: the source run's last nonzero LR")
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the plan and exit without training")
    args = parser.parse_args(argv)

    if args.additional_epochs <= 0:
        parser.error("--additional-epochs must be positive")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    signature = state["signature"]
    progress = state["progress"]
    model_cfg, train_cfg, task_cfg = configs_from_signature(signature)

    completed = int(progress["epoch"])
    if completed < int(signature["epochs"]):
        parser.error(
            f"checkpoint is mid-run ({completed}/{signature['epochs']} epochs). "
            "Use --resume_checkpoint on run_experiment.py to finish it first; "
            "this tool extends runs that already completed their budget."
        )

    peak_lr = args.peak_lr if args.peak_lr is not None else last_nonzero_lr(progress)
    if not peak_lr:
        parser.error("could not infer a peak LR from the checkpoint; pass --peak-lr")

    print(f"source run_id       : {signature.get('run_id')}")
    print(f"epochs completed    : {completed}")
    print(f"stored final LR     : "
          f"{(progress.get('completed') or {}).get('epoch_end_learning_rates', ['?'])[-1]}")
    print(f"continuation epochs : {args.additional_epochs}")
    print(f"fresh schedule peak : {peak_lr:g} (warmup {args.warmup_fraction})")
    print(f"train samples       : {task_cfg.num_samples}")
    if args.dry_run:
        print("\ndry run: nothing trained.")
        return 0

    device = torch.device(args.device)
    vocab = build_default_vocab(length=task_cfg.length, dimension=task_cfg.dimension)

    train_instances = generate_protocol_packed_instances(
        num_samples=task_cfg.num_samples, length=task_cfg.length,
        dimension=task_cfg.dimension, mod=task_cfg.mod,
        true_rate=task_cfg.true_rate, rng=random.Random(train_cfg.seed),
        generator_mode=task_cfg.generator_mode,
        corruption_rate=task_cfg.corruption_rate)
    val_instances = generate_protocol_packed_instances(
        num_samples=args.val_samples, length=task_cfg.length,
        dimension=task_cfg.dimension, mod=task_cfg.mod,
        true_rate=task_cfg.true_rate, rng=random.Random(args.eval_seed),
        generator_mode=task_cfg.generator_mode,
        corruption_rate=task_cfg.corruption_rate)
    train_ds = Task3SumDataset(
        train_instances, num_filler=task_cfg.num_filler, vocab=vocab,
        seed=train_cfg.seed, vocab_reduction=task_cfg.vocab_reduction,
        parallel_ratio=train_cfg.parallel_ratio,
        filler_ratio=train_cfg.filler_ratio,
        serial_ratio=train_cfg.serial_ratio,
        immediate_ratio=train_cfg.immediate_ratio,
        neutral_ratio=train_cfg.neutral_ratio)
    val_ds = Task3SumDataset(
        val_instances, format_type="filler", num_filler=task_cfg.num_filler,
        vocab=vocab, seed=args.eval_seed,
        vocab_reduction=task_cfg.vocab_reduction)

    model = create_model(
        replace(model_cfg, vocab_size=len(vocab)),
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab, task_cfg=task_cfg,
        compact_reduced_features=task_cfg.vocab_reduction)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        weight_decay=train_cfg.weight_decay)
    restore_optimizer(optimizer, state["optimizer_state_dict"], peak_lr, device)

    loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                        collate_fn=pad_collate_fn, num_workers=train_cfg.num_workers,
                        pin_memory=train_cfg.pin_memory and device.type == "cuda",
                        **({"persistent_workers": True,
                            "prefetch_factor": train_cfg.prefetch_factor}
                           if train_cfg.num_workers > 0 else {}))
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                            collate_fn=pad_collate_fn)
    steps_per_epoch = (len(train_ds) + train_cfg.batch_size - 1) // train_cfg.batch_size
    scheduler = build_schedule(optimizer, steps_per_epoch * args.additional_epochs,
                               args.warmup_fraction)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    ans_id = vocab.token2id["ANS"]
    true_id, false_id = vocab.token2id["True"], vocab.token2id["False"]

    accuracies: List[float] = []
    epoch_seconds: List[float] = []
    for epoch in range(args.additional_epochs):
        model.train()
        start = time.perf_counter()
        loss_sum, loss_count = 0.0, 0
        for index, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model.loss_logits(batch["input_tuples"].to(device),
                                       batch["targets"].to(device))
            loss = supervised_cross_entropy(logits, batch, criterion, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach())
            loss_count += 1
            if index and index % 2000 == 0:
                print(f"  epoch {completed + epoch + 1} step {index}/{steps_per_epoch} "
                      f"loss {loss_sum / loss_count:.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.3e} "
                      f"[{(time.perf_counter() - start) / 60:.1f} min]", flush=True)
        epoch_seconds.append(time.perf_counter() - start)
        accuracy, counts = evaluate_accuracy(
            model, val_loader, device, ans_id, true_id, false_id,
            precision=train_cfg.precision, return_prediction_counts=True)
        accuracies.append(accuracy)
        print(f"epoch {completed + epoch + 1}: filler_accuracy {accuracy:.4f} "
              f"({epoch_seconds[-1] / 3600:.2f}h)")

    report = {
        "continuation_version": CONTINUATION_VERSION,
        "is_canonical_experiment_result": False,
        "distribution_note": (
            "Continuation of a completed run under a NEW learning-rate schedule. "
            "Not a fixed-budget run and not comparable to arms that stopped at "
            "their planned budget. Do not place on an accuracy-vs-N curve."
        ),
        "source_checkpoint": str(args.checkpoint),
        "source_run_id": signature.get("run_id"),
        "source_epochs_completed": completed,
        "additional_epochs": args.additional_epochs,
        "peak_lr": peak_lr,
        "warmup_fraction": args.warmup_fraction,
        "filler_accuracy_per_added_epoch": accuracies,
        "final_filler_accuracy": accuracies[-1],
        "final_prediction_counts": counts,
        "epoch_seconds": epoch_seconds,
    }
    out = args.out or (REPO_ROOT / "results" / "continuations" /
                       f"{signature.get('run_id')}_plus{args.additional_epochs}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
