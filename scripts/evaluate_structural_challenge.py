#!/usr/bin/env python3
"""Evaluate an Experiment-0 checkpoint on the 6-stratum structural challenge set."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from exp0.checkpoint_analysis import _autocast_context, _config_from_mapping
from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import (
    PackedInstances,
    Task3SumDataset,
    build_default_vocab,
    pad_collate_fn,
)
from exp0.generation import generate_protocol_packed_instances
from exp0.train import create_model, evaluate_accuracy


def evaluate_on_challenge_instances(
    model: torch.nn.Module,
    instances_records: List[Dict[str, Any]],
    vocab,
    task_cfg: Task3SumConfig,
    device: torch.device,
    precision: str = "bf16",
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Evaluate model on structural challenge records and collect per-instance metrics."""
    # Convert records to PackedInstances
    num_instances = len(instances_records)
    tuples_arr = np.zeros((num_instances, task_cfg.length, task_cfg.dimension), dtype=np.uint8)
    has_3sum_arr = np.zeros(num_instances, dtype=bool)
    matching_indices_arr = np.full((num_instances, 3), -1, dtype=np.int16)

    for i, rec in enumerate(instances_records):
        tuples_arr[i] = np.array(rec["tuples"], dtype=np.uint8)
        has_3sum_arr[i] = rec["realized_label"]
        if rec.get("planted_matching_indices") is not None:
            matching_indices_arr[i] = np.array(rec["planted_matching_indices"], dtype=np.int16)

    packed = PackedInstances(
        tuples=tuples_arr,
        has_3sum=has_3sum_arr,
        matching_indices=matching_indices_arr,
        length=task_cfg.length,
        dimension=task_cfg.dimension,
        mod=task_cfg.mod,
    )

    dataset = Task3SumDataset(
        packed,
        format_type="filler",
        num_filler=task_cfg.num_filler,
        vocab=vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    ans_token_id = vocab.token2id["ANS"]
    ans_true_id = vocab.token2id["True"]
    ans_false_id = vocab.token2id["False"]

    model.eval()
    per_instance_results = []
    stratum_correct: Dict[str, int] = {}
    stratum_total: Dict[str, int] = {}

    idx_offset = 0
    with torch.no_grad():
        for batch in loader:
            input_tuples = batch["input_tuples"].to(device)
            targets = batch["targets"].to(device)
            has_3sum = batch["has_3sum"].to(device)
            ans_mask = targets.eq(ans_token_id)
            ans_positions = ans_mask.to(dtype=torch.int64).argmax(dim=1)

            with _autocast_context(device, precision):
                if hasattr(model, "answer_logits"):
                    answer_logits = model.answer_logits(input_tuples, targets, ans_positions)
                else:
                    logits = model(input_tuples, targets)
                    batch_idx = torch.arange(targets.shape[0], device=device)
                    answer_logits = logits[batch_idx, ans_positions]

            pred_ids = answer_logits.argmax(dim=-1)
            true_l = answer_logits[:, ans_true_id].float().cpu().tolist()
            false_l = answer_logits[:, ans_false_id].float().cpu().tolist()
            pred_list = pred_ids.cpu().tolist()
            label_list = has_3sum.cpu().tolist()

            for b in range(targets.shape[0]):
                global_idx = idx_offset + b
                rec = instances_records[global_idx]
                stratum = rec["stratum"]
                pred_bool = (pred_list[b] == ans_true_id)
                expected_bool = label_list[b]
                is_correct = (pred_bool == expected_bool)

                margin = true_l[b] - false_l[b]

                stratum_total[stratum] = stratum_total.get(stratum, 0) + 1
                if is_correct:
                    stratum_correct[stratum] = stratum_correct.get(stratum, 0) + 1

                per_instance_results.append({
                    "index": global_idx,
                    "stratum": stratum,
                    "realized_label": expected_bool,
                    "predicted_label": pred_bool,
                    "is_correct": is_correct,
                    "true_logit": true_l[b],
                    "false_logit": false_l[b],
                    "margin": margin,
                    "near_match_2of3_count": rec.get("near_match_2of3_count"),
                })
            idx_offset += targets.shape[0]

    strata_summary = {}
    for s in sorted(stratum_total):
        tot = stratum_total[s]
        corr = stratum_correct.get(s, 0)
        strata_summary[s] = {
            "total": tot,
            "correct": corr,
            "accuracy": corr / tot if tot > 0 else 0.0,
        }

    overall_corr = sum(stratum_correct.values())
    overall_tot = len(instances_records)

    return {
        "overall_accuracy": overall_corr / overall_tot if overall_tot > 0 else 0.0,
        "strata_summary": strata_summary,
        "per_instance": per_instance_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file path (.pt)")
    parser.add_argument("--challenge_set", type=Path, required=True, help="Challenge set JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True, help="Output evaluation JSON")
    args = parser.parse_args(argv)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    signature = state["signature"]
    task_cfg = _config_from_mapping(Task3SumConfig, signature["task"], "task")
    model_cfg = _config_from_mapping(ModelConfig, signature["model"], "model")
    train_cfg = _config_from_mapping(TrainConfig, signature["training"], "training")

    vocab = build_default_vocab(length=task_cfg.length, dimension=task_cfg.dimension)
    model = create_model(
        model_cfg,
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab,
        task_cfg=task_cfg,
        compact_reduced_features=task_cfg.vocab_reduction,
    )
    model.load_state_dict(state["model_state_dict"])
    device = torch.device(args.device)
    model = model.to(device)

    # 1. Canonical validation accuracy (eval_seed=9999, val_samples=2000)
    val_instances = generate_protocol_packed_instances(
        num_samples=2000,
        length=task_cfg.length,
        dimension=task_cfg.dimension,
        mod=task_cfg.mod,
        true_rate=task_cfg.true_rate,
        rng=random.Random(9999),
        generator_mode=task_cfg.generator_mode,
        corruption_rate=task_cfg.corruption_rate,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        num_filler=task_cfg.num_filler,
        vocab=vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)
    canonical_acc, counts = evaluate_accuracy(
        model,
        val_loader,
        device,
        vocab.token2id["ANS"],
        vocab.token2id["True"],
        vocab.token2id["False"],
        precision=args.precision,
        return_prediction_counts=True,
    )

    # 2. Structural challenge set evaluation
    challenge_payload = json.loads(args.challenge_set.read_text(encoding="utf-8"))
    challenge_results = evaluate_on_challenge_instances(
        model=model,
        instances_records=challenge_payload["instances"],
        vocab=vocab,
        task_cfg=task_cfg,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )

    output = {
        "checkpoint": str(args.checkpoint),
        "run_id": signature.get("run_id"),
        "seed": train_cfg.seed,
        "num_filler": task_cfg.num_filler,
        "epochs": signature.get("epochs"),
        "canonical_validation": {
            "eval_seed": 9999,
            "samples": 2000,
            "accuracy": canonical_acc,
            "counts": counts,
        },
        "structural_challenge": {
            "challenge_id": challenge_payload["provenance"]["challenge_id"],
            "content_sha256": challenge_payload["provenance"]["content_sha256"],
            "overall_accuracy": challenge_results["overall_accuracy"],
            "strata_summary": challenge_results["strata_summary"],
            "per_instance": challenge_results["per_instance"],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {args.checkpoint.name} (seed {train_cfg.seed}, N={task_cfg.num_filler}):")
    print(f"  Canonical Accuracy: {canonical_acc:6.2%}")
    print(f"  Challenge Accuracy: {challenge_results['overall_accuracy']:6.2%}")
    print("  Challenge Strata:")
    for s, data in challenge_results["strata_summary"].items():
        print(f"    {s:36}: {data['accuracy']:6.2%} ({data['correct']}/{data['total']})")
    print(f"Saved evaluation to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
