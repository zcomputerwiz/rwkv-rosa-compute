"""Benchmark AdamW implementations at the 0B parameter count.

The step profile shows the optimizer is a fixed per-parameter cost - 30.5 ms in
both the padded and grouped paths - which grouping cannot reduce, so it rose
from 9.7% to 16.0% of step time once grouping shrank everything else.

A bandwidth estimate suggests it should be far cheaper than that: AdamW touches
param, grad, exp_avg and exp_avg_sq, about 1.75 GB per step at 109.5M
parameters, which is roughly 6-9 ms at this card's achievable bandwidth.

This times the three implementations on the real parameter shapes. It measures
the optimizer alone, with gradients pre-populated, so nothing else is in frame.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from exp0.config import ModelConfig, Task3SumConfig  # noqa: E402
from exp0.dataset import build_default_vocab  # noqa: E402
from exp0.train import create_model  # noqa: E402

SHAPE = {"hidden": 768, "layers": 12, "heads": 12}


def build_model(device):
    task_cfg = Task3SumConfig(num_filler=0)
    vocab = build_default_vocab(
        length=task_cfg.length, dimension=task_cfg.dimension, mod=task_cfg.mod
    )
    torch.manual_seed(1234)
    model = create_model(
        ModelConfig(
            architecture="rwkv",
            hidden_size=SHAPE["hidden"],
            num_hidden_layers=SHAPE["layers"],
            num_attention_heads=SHAPE["heads"],
            intermediate_size=SHAPE["hidden"] * 4,
            head_dim=64,
            rwkv_kernel="reference",
            device="cuda",
            vocab_size=len(vocab),
            output_vocab_size=32000,
        ),
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab,
        task_cfg=task_cfg,
    )
    return model.to(device)


def measure(label: str, model, steps: int, **adam_kwargs) -> Dict[str, float]:
    # Fresh gradients and fresh optimizer state per variant, so no variant
    # benefits from another's warm allocator or populated state.
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, **adam_kwargs)

    for _ in range(5):
        optimizer.step()
    torch.cuda.synchronize()

    per_step: List[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        per_step.append(time.perf_counter() - start)

    del optimizer
    for parameter in model.parameters():
        parameter.grad = None
    torch.cuda.empty_cache()
    return {
        "label": label,
        "median_ms": statistics.median(per_step) * 1e3,
        "stdev_ms": statistics.stdev(per_step) * 1e3 if len(per_step) > 1 else 0.0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    model = build_model(device)
    params = sum(p.numel() for p in model.parameters())
    traffic_gb = params * 4 * 4 / 1e9

    print(f"device     : {torch.cuda.get_device_name(0)}")
    print(f"parameters : {params / 1e6:.2f} M")
    print(f"AdamW traffic per step: {traffic_gb:.2f} GB "
          "(param, grad, exp_avg, exp_avg_sq)")
    print()

    variants = [
        ("foreach (default)", {"foreach": True}),
        ("fused", {"fused": True}),
        ("single-tensor", {"foreach": False}),
    ]
    results = []
    for label, kwargs in variants:
        try:
            results.append(measure(label, model, args.steps, **kwargs))
        except Exception as exc:
            print(f"{label}: unavailable ({str(exc).splitlines()[0][:90]})")

    baseline = next((r["median_ms"] for r in results
                     if r["label"] == "foreach (default)"), None)
    print(f"  {'variant':20} {'median ms':>10} {'stdev':>8} {'speedup':>8} "
          f"{'GB/s':>8}")
    for r in results:
        speedup = baseline / r["median_ms"] if baseline else float("nan")
        bandwidth = traffic_gb / (r["median_ms"] / 1e3)
        print(f"  {r['label']:20} {r['median_ms']:>10.2f} {r['stdev_ms']:>8.2f} "
              f"{speedup:>7.3f}x {bandwidth:>8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
