#!/usr/bin/env python3
"""Measure training-throughput levers and their numerical cost, together.

Every lever here (TF32, BF16, torch.compile) is a numerical or execution
intervention that changes run identity, so a speedup alone is not actionable.
Each is reported with the loss deviation it causes.

Numerics are measured on a SINGLE forward/backward from a fixed model and batch.
Measuring them through many training steps on one batch does not work: tiny
differences amplify chaotically and the run-to-run floor swamps the effect. A
single pass is bitwise reproducible on both architectures, so the deviations
below are signal, not noise.

    python scripts/benchmark_training_precision.py --arch llama
    python scripts/benchmark_training_precision.py --arch rwkv --steps 25

Requires CUDA. torch.compile additionally requires a working Triton; see
docs/experiment0_precision_and_compile.md for the Windows toolchain constraint.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from exp0.config import ModelConfig, Task3SumConfig  # noqa: E402
from exp0.dataset import (  # noqa: E402
    Task3SumDataset,
    build_default_vocab,
    pad_collate_fn,
)
from exp0.generation import generate_protocol_packed_instances  # noqa: E402
from exp0.train import create_model  # noqa: E402

CRITERION = torch.nn.CrossEntropyLoss(ignore_index=-100)

SHAPES: Dict[str, Dict[str, Any]] = {
    # The shapes of the actual Experiment 0 runs, so numbers transfer directly.
    "llama": {"hidden": 384, "layers": 4, "heads": 6, "kernel": "reference",
              "batch": 384},
    "rwkv": {"hidden": 768, "layers": 12, "heads": 12, "kernel": "cuda",
             "batch": 128},
}


def set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def make_batch(vocab, batch_size: int, length: int, dimension: int, num_filler: int,
               device: torch.device) -> Dict[str, Any]:
    packed = generate_protocol_packed_instances(
        batch_size, length=length, dimension=dimension, rng=random.Random(7))
    dataset = Task3SumDataset(packed, num_filler=num_filler, vocab=vocab,
                              parallel_ratio=0.0, filler_ratio=1.0)
    batch = pad_collate_fn([dataset[i] for i in range(len(dataset))])
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_model(arch: str, vocab, task_cfg: Task3SumConfig, device: torch.device):
    shape = SHAPES[arch]
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    model = create_model(
        ModelConfig(architecture=arch, hidden_size=shape["hidden"],
                    num_hidden_layers=shape["layers"],
                    num_attention_heads=shape["heads"],
                    intermediate_size=shape["hidden"] * 4, head_dim=64,
                    rwkv_kernel=shape["kernel"], device="cuda",
                    vocab_size=len(vocab), output_vocab_size=32000),
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab, task_cfg=task_cfg)
    return model.to(device)


def forward_backward(forward_fn: Callable, model, batch, precision: str) -> float:
    model.zero_grad(set_to_none=True)
    context = (torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16"
               else torch.autocast("cuda", enabled=False))
    with context:
        logits = forward_fn(batch["input_tuples"], batch["targets"])
        # loss_mask, not targets: targets is padded with the PAD id because it
        # is fed to the model, while loss_mask is padded with -100 so padded
        # positions are ignored by cross entropy.
        loss = CRITERION(logits.reshape(-1, logits.size(-1)).float(),
                         batch["loss_mask"][:, 1:].reshape(-1))
    loss.backward()
    return float(loss.detach())


def compiled_or_plain(model, compiled: bool) -> Callable:
    """Compile the bound method that is actually invoked.

    torch.compile(model) wraps forward only, and OptimizedModule.__getattr__
    forwards every other attribute to the original module -- so compiling the
    module and then calling .loss_logits() silently runs eager and reports a
    misleading 1.00x.
    """
    return torch.compile(model.loss_logits) if compiled else model.loss_logits


VARIANTS: Dict[str, List[Tuple[str, str, bool, bool]]] = {
    # label, precision, tf32, compiled
    "llama": [
        ("fp32", "fp32", False, False),
        ("fp32+TF32", "fp32", True, False),
        ("bf16", "bf16", False, False),
        ("bf16+compile", "bf16", False, True),
    ],
    "rwkv": [
        ("bf16", "bf16", False, False),
        ("bf16+compile", "bf16", False, True),
    ],
}


def run(arch: str, steps: int, repeats: int) -> None:
    device = torch.device("cuda")
    shape = SHAPES[arch]
    task_cfg = Task3SumConfig(length=6, dimension=3, num_filler=36)
    vocab = build_default_vocab(length=6, dimension=3)
    batch = make_batch(vocab, shape["batch"], 6, 3, 36, device)

    print(f"\n=== {arch} (batch {shape['batch']}, "
          f"{shape['layers']} layers, hidden {shape['hidden']}) ===")
    print(f"  {'variant':22} {'ms/step':>9} {'speedup':>8} {'loss':>14} "
          f"{'d(loss)':>12} {'floor':>10}")

    baseline_ms: Optional[float] = None
    baseline_loss: Optional[float] = None
    for label, precision, tf32, compiled in VARIANTS[arch]:
        set_tf32(tf32)
        model = make_model(arch, vocab, task_cfg, device)
        forward_fn = compiled_or_plain(model, compiled)
        try:
            losses = [forward_backward(forward_fn, model, batch, precision)
                      for _ in range(repeats)]
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            def training_step() -> None:
                optimizer.zero_grad(set_to_none=True)
                forward_backward(forward_fn, model, batch, precision)
                optimizer.step()

            for _ in range(5):
                training_step()
            torch.cuda.synchronize()
            times = []
            for _ in range(steps):
                start = time.perf_counter()
                training_step()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        except Exception as exc:  # a lever that cannot run is a result
            print(f"  {label:22} FAILED: {type(exc).__name__}: {str(exc)[:48]}")
            model = None
            torch.cuda.empty_cache()
            continue

        median_ms = statistics.median(times) * 1000
        loss = statistics.mean(losses)
        floor = max(losses) - min(losses)
        if baseline_ms is None:
            baseline_ms, baseline_loss = median_ms, loss
            speed, delta = "1.00x", "-"
        else:
            speed = f"{baseline_ms / median_ms:.2f}x"
            delta = f"{loss - baseline_loss:+.3e}"
        print(f"  {label:22} {median_ms:8.2f} {speed:>8} {loss:14.8f} "
              f"{delta:>12} {floor:10.1e}")
        model = None
        forward_fn = None
        torch.cuda.empty_cache()

    print("\n  floor is the spread across identical repeats: 0.0e+00 means the "
          "pass is\n  bitwise reproducible, so d(loss) is signal rather than "
          "noise.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=("llama", "rwkv", "both"), default="both")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        parser.error("CUDA is required for this benchmark.")
    for arch in (("llama", "rwkv") if args.arch == "both" else (args.arch,)):
        run(arch, args.steps, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
