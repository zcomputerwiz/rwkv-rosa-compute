#!/usr/bin/env python3
"""Evaluation-only validation of the batched QSA indexer on the D=1 gate cell.

Authorized by `REPLY_AND_TASK_ADA_QWEN4_BATCHED_QSA_INDEXER_SHANNON.md`. This
runs no training, writes no artifact under `results/`, and reads the canonical
checkpoints without modifying them.

Part A loads the canonical final seed-3011 checkpoint and evaluates the complete
20,480-instance training bank twice, once through the preserved upstream indexer
and once through the batched one, requiring exact equality of every selected
mask, logit, prediction, correct count and accuracy.

Part B, only if Part A is exact, scores the seeds 3012 and 3013 final
checkpoints on the same bank as a post-hoc fit-versus-generalize diagnostic. It
cannot change the D=1 FAIL verdict.

Part C reports CUDA-event timings, remaining synchronizations, peak memory and
utilization at the registered B=64, T=18 shape.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer

from exp0.config import TrainConfig
from exp1.dataset import PointerChaseDataset
from exp1.pointer_chase import ChaseSpec, generate_dataset
from exp1.qwen4_micro import Qwen4MicroConfig, create_qwen4_micro_model
from exp1.train import evaluate_vway_accuracy, forward_logits
from rosa_compute.diagnostics import get_artifact_environment

GATE = dict(num_nodes=16, num_maps=4, max_depth=32, depth=1,
            queries_per_memory=4, train_memories=5120, train_data_seed=3014,
            batch_size=64)
CANONICAL_HELD_OUT = {3011: 0.06396484375, 3012: 0.0751953125,
                      3013: 0.06884765625}


def checkpoint_path(seed: int) -> Path:
    root = (Path(r"D:\GitHub\rwkv-rosa-compute") if seed == 3011
            else Path(r"D:\GitHub\rwkv-rosa-compute-gate"))
    return root / "results" / "qwen4_population_ckpt" / "d1" / f"seed{seed}" / "latest.pt"


def build_bank():
    spec = ChaseSpec(num_nodes=GATE["num_nodes"], num_maps=GATE["num_maps"],
                     max_depth=GATE["max_depth"])
    instances = generate_dataset(
        GATE["train_memories"], queries_per_memory=GATE["queries_per_memory"],
        depth=GATE["depth"], seed=GATE["train_data_seed"],
        num_nodes=GATE["num_nodes"], num_maps=GATE["num_maps"],
    )
    return spec, PointerChaseDataset(instances, spec, num_silent=0,
                                     silent_kind=None, neutral_vector=None)


def build_model(spec, seed: int, device, batched: bool):
    """Load a canonical checkpoint into a hybrid model on the chosen path."""
    config = Qwen4MicroConfig(vocab_size=GATE["num_nodes"])
    model = create_qwen4_micro_model(config, d_input=spec.d_input)
    if not batched:
        removed = 0
        for module in model.modules():
            if isinstance(module, Qwen4ExpTextQSAIndexer) and "forward" in module.__dict__:
                del module.__dict__["forward"]
                removed += 1
        assert removed == 1, f"expected one indexer override to remove, got {removed}"
    else:
        assert any("forward" in m.__dict__ for m in model.modules()
                   if isinstance(m, Qwen4ExpTextQSAIndexer))

    blob = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
    report = model.load_state_dict(blob["model_state_dict"], strict=True)
    assert not report.missing_keys and not report.unexpected_keys
    return model.to(device).eval(), blob["signature"]


def indexer_of(model):
    found = [m for m in model.modules() if isinstance(m, Qwen4ExpTextQSAIndexer)]
    assert len(found) == 1
    return found[0]


def sweep(model, dataset, spec, train_cfg, device):
    """One full-bank pass, hashing every mask, logit and prediction."""
    from exp1.dataset import exp1_collate_fn
    from exp1.train import _create_loader

    masks, logits_digest, preds = (hashlib.sha256() for _ in range(3))
    captured = []

    def hook(_module, _args, output):
        captured.append(output)

    handle = indexer_of(model).register_forward_hook(hook)
    loader = _create_loader(dataset, train_cfg, device, shuffle=False)
    loader.collate_fn = exp1_collate_fn

    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    try:
        with torch.no_grad():
            for batch in loader:
                captured.clear()
                inputs = batch["input_tuples"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                out = forward_logits(model, inputs)
                assert len(captured) == 1
                mask = captured[0]
                masks.update(mask.detach().to(torch.uint8).cpu().numpy().tobytes())
                logits_digest.update(out.detach().float().cpu().numpy().tobytes())
                prediction = out[:, : spec.num_nodes].argmax(dim=-1)
                preds.update(prediction.detach().to(torch.int32).cpu().numpy().tobytes())
                correct += prediction.eq(targets).sum()
                total += targets.shape[0]
    finally:
        handle.remove()
    return {
        "selected_mask_sha256": masks.hexdigest(),
        "logits_sha256": logits_digest.hexdigest(),
        "predictions_sha256": preds.hexdigest(),
        "correct": int(correct.item()),
        "total": total,
        "accuracy": correct.item() / total,
    }


def sync_count(model, inputs):
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            forward_logits(model, inputs)
    torch.cuda.set_sync_debug_mode("default")
    return sum(1 for w in caught if "synchroniz" in str(w.message))


def timed_forward(model, inputs, repeats=30):
    with torch.no_grad():
        for _ in range(5):
            forward_logits(model, inputs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        with torch.no_grad():
            forward_logits(model, inputs)
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


class Utilization(threading.Thread):
    """Poll nvidia-smi while a pass runs; the medians go in the report."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.wait(0.25):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                util, power = out.split(",")
                self.samples.append((float(util), float(power)))
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="comparison record; must not be under results/")
    args = parser.parse_args()
    assert "results" not in args.out.parts, "refusing to write under results/"

    device = torch.device("cuda")
    environment = get_artifact_environment()
    print(json.dumps({"environment": environment}, indent=2, default=str)[:900])

    spec, bank = build_bank()
    print(f"\ntraining bank: {len(bank)} instances, T={bank[0]['input_tuples'].shape[0]}")
    assert len(bank) == 20480, len(bank)

    train_cfg = TrainConfig(seed=3011, batch_size=GATE["batch_size"],
                            precision="fp32", num_workers=0, epochs=10)

    record = {"gate": GATE, "environment": environment, "arms": {}}

    # ---- Part A: exact full-bank equivalence on seed 3011 -------------------
    print("\n=== Part A: seed 3011 full-bank equivalence ===")
    for name, batched in (("upstream", False), ("batched", True)):
        model, signature = build_model(spec, 3011, device, batched)
        started = time.perf_counter()
        result = sweep(model, bank, spec, train_cfg, device)
        result["wall_seconds"] = round(time.perf_counter() - started, 2)
        result["evaluate_vway_accuracy"] = evaluate_vway_accuracy(
            model, bank, train_cfg, device)
        record["arms"][name] = result
        record.setdefault("checkpoint_signature", signature)
        print(f"  {name:<9} acc={result['accuracy']!r} "
              f"correct={result['correct']}/{result['total']} "
              f"{result['wall_seconds']}s")
        for key in ("selected_mask_sha256", "logits_sha256", "predictions_sha256"):
            print(f"      {key} {result[key]}")
        del model
        torch.cuda.empty_cache()

    up, ba = record["arms"]["upstream"], record["arms"]["batched"]
    checks = {
        "selected masks identical": up["selected_mask_sha256"] == ba["selected_mask_sha256"],
        "logits identical": up["logits_sha256"] == ba["logits_sha256"],
        "predictions identical": up["predictions_sha256"] == ba["predictions_sha256"],
        "correct counts identical": up["correct"] == ba["correct"],
        "accuracy identical": up["accuracy"] == ba["accuracy"],
        "canonical evaluator agrees": (
            up["evaluate_vway_accuracy"] == ba["evaluate_vway_accuracy"]
            == ba["accuracy"]),
    }
    record["part_a"] = checks
    print()
    for label, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}")
    exact = all(checks.values())

    # ---- Part B: fit-versus-generalize diagnostic ---------------------------
    record["part_b"] = {}
    if exact:
        print("\n=== Part B: training-bank accuracy, batched path ===")
        print(f"  {'seed':<6}{'train bank':>14}{'held out (canonical)':>24}")
        for seed in (3011, 3012, 3013):
            model, _ = build_model(spec, seed, device, batched=True)
            accuracy = evaluate_vway_accuracy(model, bank, train_cfg, device)
            record["part_b"][str(seed)] = {
                "train_bank_accuracy": accuracy,
                "held_out_final": CANONICAL_HELD_OUT[seed],
            }
            print(f"  {seed:<6}{accuracy:>14.10f}{CANONICAL_HELD_OUT[seed]:>24.10f}")
            del model
            torch.cuda.empty_cache()
    else:
        print("\nPart A was not exact; Part B is skipped by the task's condition.")

    # ---- Part C: performance ------------------------------------------------
    print("\n=== Part C: performance at B=64, T=18 ===")
    inputs = torch.stack([bank[i]["input_tuples"] for i in range(64)]).to(device)
    record["part_c"] = {}
    for name, batched in (("upstream", False), ("batched", True)):
        model, _ = build_model(spec, 3011, device, batched)
        torch.cuda.reset_peak_memory_stats()
        median = timed_forward(model, inputs)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        reserved = torch.cuda.max_memory_reserved() / 2 ** 20
        syncs = sync_count(model, inputs)

        watcher = Utilization()
        watcher.start()
        started = time.perf_counter()
        evaluate_vway_accuracy(model, bank, train_cfg, device)
        pass_seconds = time.perf_counter() - started
        watcher.stop.set()
        watcher.join(timeout=2)
        util = [s[0] for s in watcher.samples] or [float("nan")]
        power = [s[1] for s in watcher.samples] or [float("nan")]

        record["part_c"][name] = {
            "median_forward_ms": round(median, 3),
            "syncs_per_forward": syncs,
            "peak_allocated_mib": round(peak, 1),
            "peak_reserved_mib": round(reserved, 1),
            "full_bank_eval_seconds": round(pass_seconds, 2),
            "instances_per_second": round(len(bank) / pass_seconds, 1),
            "median_gpu_utilization_percent": statistics.median(util),
            "median_power_w": statistics.median(power),
        }
        print(f"  {name:<9} fwd {median:8.2f} ms   syncs {syncs:>5}   "
              f"peak {peak:7.1f} MiB   bank {pass_seconds:6.1f} s   "
              f"{len(bank) / pass_seconds:7.1f} inst/s   "
              f"util {statistics.median(util):.0f}%  {statistics.median(power):.0f} W")
        del model
        torch.cuda.empty_cache()

    speedup = (record["part_c"]["upstream"]["median_forward_ms"]
               / record["part_c"]["batched"]["median_forward_ms"])
    record["part_c"]["forward_speedup"] = round(speedup, 3)
    print(f"\n  forward speedup {speedup:.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"\nrecord written to {args.out}")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
