"""Tiny overfit gate for Experiment 1 (H2).

The plan's gate before any production H2 run: make RWKV memorize a small set at
several depths, with both insufficient and sufficient N. If it cannot fit 256
examples, the task or the training setup is wrong and no production compute will
rescue it.

**This is a capacity check, not a science run.** It deliberately trains on the
evaluation set - memorization is the thing being tested. Nothing here produces a
`run_id` or belongs on any curve.

Uses the RWKV7Backbone directly rather than `InputEmbedWrapper`, because the
Experiment-0 wrapper is built for next-token prediction over
`(input_tuples, target_ids)` while H2 encodes one tensor per instance and reads
a single answer position. Same backbone, task-appropriate head.

**Runs without the CUDA JIT toolchain.** Pass ``--rwkv_kernel reference`` for the
pure-PyTorch recurrence; far slower than the fused kernel, and irrelevant at
these sizes.

Example::

    python scripts/exp1_overfit_gate.py --examples 256 --depths 2 8 --silent 0 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from exp0.models.rwkv import RWKV7Backbone  # noqa: E402
from exp1.encoding import EncodingSpec, encode_batch  # noqa: E402
from exp1.sequential_task import generate_dataset  # noqa: E402


class SequentialTaskModel(nn.Module):
    """RWKV backbone, linear input projection, answer head at the final position."""

    def __init__(self, d_input: int, num_classes: int, *, hidden_size: int,
                 num_layers: int, intermediate_size: int, head_dim: int,
                 rwkv_kernel: str) -> None:
        super().__init__()
        self.proj = nn.Linear(d_input, hidden_size)
        self.backbone = RWKV7Backbone(
            hidden_size=hidden_size,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            rwkv_kernel=rwkv_kernel,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(self.proj(x))
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.head(hidden[:, -1, :])  # the query position


def run_one(*, examples: int, depth: int, num_silent: int, args,
            device: torch.device) -> dict:
    silent_kind = None if num_silent == 0 else args.silent_kind
    data = generate_dataset(
        examples, [depth], seed=args.seed,
        num_instructions=args.instructions,
        num_registers=args.registers, mod=args.mod,
    )
    spec = EncodingSpec(num_registers=args.registers, mod=args.mod,
                        num_instructions=args.instructions,
                        num_silent=num_silent)
    inputs, answers, _ = encode_batch(data, spec, silent_kind=silent_kind)
    inputs, answers = inputs.to(device), answers.to(device)

    torch.manual_seed(args.seed)
    model = SequentialTaskModel(
        spec.d_input, args.mod,
        hidden_size=args.hidden_size, num_layers=args.layers,
        intermediate_size=args.hidden_size * 4, head_dim=args.head_dim,
        rwkv_kernel=args.rwkv_kernel,
    ).to(device)
    if args.precision == "bf16":
        model = model.to(torch.bfloat16)
        inputs = inputs.to(torch.bfloat16)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    start = time.perf_counter()
    best = 0.0
    for step in range(1, args.steps + 1):
        perm = torch.randperm(inputs.shape[0], device=device)
        for i in range(0, inputs.shape[0], args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(inputs[idx]).float(), answers[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                acc = 0.0
                for i in range(0, inputs.shape[0], args.batch_size):
                    logits = model(inputs[i:i + args.batch_size])
                    acc += (logits.argmax(-1) == answers[i:i + args.batch_size]).sum().item()
                acc /= inputs.shape[0]
            best = max(best, acc)
            model.train()
            print(f"    step {step:5d}  loss {float(loss.detach()):7.4f}  train acc {100*acc:6.2f}%",
                  flush=True)
            if acc >= args.target:
                break
    return {
        "examples": examples, "depth": depth, "num_silent": num_silent,
        "silent_kind": silent_kind, "steps_run": step,
        "best_train_accuracy": best,
        "memorized": best >= args.target,
        "seconds": round(time.perf_counter() - start, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--examples", type=int, nargs="+", default=[256])
    p.add_argument("--depths", type=int, nargs="+", default=[2, 8])
    p.add_argument("--silent", type=int, nargs="+", default=[0, 16],
                   help="N values; 0 is arm A")
    p.add_argument("--silent_kind", default="scratchpad",
                   choices=["scratchpad", "neutral"])
    p.add_argument("--instructions", type=int, default=32)
    p.add_argument("--registers", type=int, default=4)
    p.add_argument("--mod", type=int, default=13)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--rwkv_kernel", default="reference",
                   choices=["reference", "cuda"],
                   help="reference needs no MSVC/CUDA toolchain")
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--target", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"device {device} | kernel {args.rwkv_kernel} | precision {args.precision}")
    print(f"model  hidden {args.hidden_size} layers {args.layers} "
          f"head_dim {args.head_dim}")
    print(f"chance {100/args.mod:.2f}%  target {100*args.target:.0f}%\n")

    results = []
    for n in args.examples:
        for depth in args.depths:
            for silent in args.silent:
                print(f"  examples={n} depth={depth} N={silent}")
                r = run_one(examples=n, depth=depth, num_silent=silent,
                            args=args, device=device)
                results.append(r)
                verdict = "MEMORIZED" if r["memorized"] else "FAILED"
                print(f"    -> {verdict} best {100*r['best_train_accuracy']:.2f}% "
                      f"in {r['steps_run']} steps, {r['seconds']}s\n", flush=True)

    print("summary")
    print(f"  {'examples':>8s} {'depth':>5s} {'N':>4s} {'best acc':>9s}  verdict")
    for r in results:
        print(f"  {r['examples']:8d} {r['depth']:5d} {r['num_silent']:4d} "
              f"{100*r['best_train_accuracy']:8.2f}%  "
              f"{'MEMORIZED' if r['memorized'] else 'FAILED'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"config": vars(args) | {"out": str(args.out), "device": str(args.device)},
             "results": results}, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    failed = [r for r in results if not r["memorized"]]
    if failed:
        print(f"\n{len(failed)} configuration(s) did not memorize. "
              "Do not launch production training until this is understood.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
