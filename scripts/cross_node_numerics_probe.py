"""Measure whether the RWKV-7 bf16 forward pass is bit-identical across GPUs.

The 0B seed study evaluated its two arms on two different GPUs - N=0 on an
RTX 4060 Ti, N=36 on an RTX 3070 - so arm and hardware are confounded. The
pre-registration discloses this as a limitation it cannot remove. This probe
bounds it.

We already know evaluation AUC moves by about 0.003 when the *batch size*
changes at fixed hardware, because the bf16 kernel tiles differently. Tiling
and GEMM algorithm selection depend on tensor shapes and on the SM count of
the device, not on the values in the weights. So running identical weights and
identical inputs on two devices and comparing the raw logits answers the
question directly: if the bytes match, the hardware term is exactly zero and no
trained checkpoint could differ either.

Nothing needs to be transferred between nodes. The weights are generated from
a fixed seed on CPU and then moved to the device, and the inputs come from the
same deterministic generator the evaluator uses, so both nodes construct the
same tensors from the same commit.

Run on each node and compare `logits_sha256`:

    python scripts/cross_node_numerics_probe.py --out probe_<node>.json

Identical digests   the confound is zero; record the bound and move on.
Differing digests   there is a real hardware term; `max_abs_delta` in a
                    pairwise comparison sizes it, and the study's limitation
                    section needs that number rather than an assurance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exp0.checkpoint_analysis import _config_from_mapping  # noqa: E402
from exp0.config import ModelConfig, Task3SumConfig  # noqa: E402
from exp0.dataset import (  # noqa: E402
    Task3SumDataset,
    build_default_vocab,
    pad_collate_fn,
)
from exp0.diagnostics import _autocast_context  # noqa: E402
from exp0.generation import generate_protocol_packed_instances  # noqa: E402
from exp0.train import create_model  # noqa: E402

# The 0B study's recorded signature, copied verbatim from
# n0/checkpoints/d6d23abcab7a898b/seed_44/epoch_005.pt and built through the
# same _config_from_mapping the evaluator uses. Spelled out rather than read
# from a checkpoint so this runs on a node holding no study checkpoints, and
# so the two nodes provably construct the same model.
STUDY_TASK = {
    "length": 6, "dimension": 3, "mod": 10, "num_filler": 0, "true_rate": 0.5,
    "vocab_reduction": True, "include_separator_token": True,
    "include_eos_target": True, "generator_mode": "source_corrupted",
    "corruption_rate": 1.3333333333333333, "seed": 42, "num_samples": 2000000,
}
STUDY_MODEL = {
    "architecture": "rwkv", "init_mode": "pretrained",
    "rwkv_checkpoint_sha256":
        "e10d7b1930c2644c5c6b194444774d6d82ec8212a78763493149de09aac7d83f",
    "rwkv_kernel": "cuda", "hidden_size": 768, "num_hidden_layers": 12,
    "num_attention_heads": 12, "intermediate_size": 3072, "head_dim": 64,
    "llama_rope_theta": 10000.0, "llama_initializer_range": 0.02,
    "match3_shared_input_features": True, "vocab_size": 1145,
    "output_vocab_size": 32000, "device": "cuda",
}
# Weights are randomly initialized from this seed on CPU, not loaded from the
# pretrained file. What is being measured is whether the kernel and GEMMs
# produce identical bytes for identical inputs, and tiling depends on shapes
# and SM count rather than on weight values, so real weights are unnecessary -
# and requiring them would reintroduce a file dependency.
WEIGHT_SEED = 20260827
DATA_SEED = 9999
BATCH_SIZE = 128


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = ap.parse_args(argv)

    task_cfg = _config_from_mapping(Task3SumConfig, STUDY_TASK, "task")
    model_cfg = _config_from_mapping(ModelConfig, STUDY_MODEL, "model")
    vocab = build_default_vocab(length=task_cfg.length, dimension=task_cfg.dimension)

    # Single-threaded, because torch's CPU RNG fills large tensors through
    # at::parallel_for and the values therefore depend on the thread count.
    # v1 of this probe did not pin it, so two machines with different core
    # counts built different weights while believing they had not.
    torch.set_num_threads(1)
    torch.manual_seed(WEIGHT_SEED)
    model = create_model(
        model_cfg,
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab,
        task_cfg=task_cfg,
        compact_reduced_features=task_cfg.vocab_reduction,
    )

    # Overwrite every floating tensor from one seeded generator. Construction
    # alone is not enough: RWKV-7 initializes the low-rank time-mix pairs with
    # one side at zero, so w1 @ w2 is zero whatever w2 holds and the whole LoRA
    # path is inert on a freshly built model. v1 hashed logits from that model
    # and therefore never exercised the low-rank products - the part of the
    # computation most likely to differ between architectures. Refilling both
    # sides makes every branch active.
    gen = torch.Generator().manual_seed(WEIGHT_SEED)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.copy_(torch.empty(tensor.shape, dtype=torch.float32)
                             .normal_(0.0, 0.02, generator=gen).to(tensor.dtype))
    model.eval()
    param_sha = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        param_sha.update(name.encode())
        param_sha.update(p.detach().cpu().contiguous().numpy().tobytes())

    device = torch.device(args.device)
    model = model.to(device)

    instances = generate_protocol_packed_instances(
        num_samples=args.batch_size,
        length=task_cfg.length,
        dimension=task_cfg.dimension,
        mod=task_cfg.mod,
        true_rate=task_cfg.true_rate,
        rng=random.Random(DATA_SEED),
        generator_mode=task_cfg.generator_mode,
        corruption_rate=task_cfg.corruption_rate,
    )
    ds = Task3SumDataset(instances, format_type="filler",
                         num_filler=task_cfg.num_filler, vocab=vocab,
                         vocab_reduction=task_cfg.vocab_reduction)
    batch = pad_collate_fn([ds[i] for i in range(len(ds))])
    inputs = batch["input_tuples"].to(device)
    targets = batch["targets"].to(device)

    with torch.no_grad(), _autocast_context(device, args.precision):
        logits = model(inputs, targets)

    # Identity of everything that feeds the digest, so a reproducer can tell a
    # hardware difference from a stale checkout or different inputs. Without
    # these, input and code identity are asserted by the run procedure rather
    # than proved by the artifact.
    input_sha = hashlib.sha256(inputs.detach().cpu().contiguous().numpy().tobytes())
    input_sha.update(targets.detach().cpu().contiguous().numpy().tobytes())
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        import subprocess
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]),
                                 "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10)
        commit = commit.stdout.strip() if commit.returncode == 0 else "UNAVAILABLE"
    except Exception:
        commit = "UNAVAILABLE"

    flat = logits.detach().float().cpu().contiguous()
    if not torch.isfinite(flat).all():
        raise SystemExit("non-finite logits; the probe is not measuring anything")
    digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()

    payload = {
        "probe_version": 3,
        "logits_sha256": digest,
        "param_sha256": param_sha.hexdigest(),
        "logits_shape": list(flat.shape),
        "input_sha256": input_sha.hexdigest(),
        "script_sha256": script_sha,
        "commit": commit,
        # A few exact values, so two artifacts can be compared by eye and a
        # difference can be sized without re-running anything.
        "logits_sample": [round(float(v), 8) for v in flat.flatten()[:8]],
        "logits_sum": float(flat.double().sum()),
        "settings": {"precision": args.precision, "batch_size": args.batch_size,
                     "device": device.type, "weight_seed": WEIGHT_SEED,
                     "data_seed": DATA_SEED, "num_threads": torch.get_num_threads()},
        "model": STUDY_MODEL, "task": STUDY_TASK,
        "environment": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"logits_sha256 {digest}")
    print(f"param_sha256  {payload['param_sha256']}")
    print(f"gpu           {payload['environment']['gpu']} "
          f"sm_{''.join(str(x) for x in (payload['environment']['capability'] or []))}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
