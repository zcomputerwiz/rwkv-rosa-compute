#!/usr/bin/env python3
"""Map where the pointer chase stops being memorised and starts being solved.

Gate 0a returned chance on all three seeds. Two distinct failures sit behind
that, and conflating them was the first mistake this tool was used to make:

  At the gate's own inherited learning rate of 1e-4, training barely started --
  loss moved 2.800 to 2.653 against ln(16) = 2.7726. That is under-training,
  not memorisation.

  At the learning rates the published literature uses, from 2e-4 up, training
  works and the model reaches 0.62 training accuracy against 0.07 held out. It
  memorises the training memories instead of learning in-context retrieval.

So this tool characterises the regime where optimisation succeeds. It does not
explain the registered gate's failure, and no report may read it that way.

Held-out accuracy alone cannot tell "has not learned" from "has memorised", and
here that distinction is the entire result. This tool always reports both, plus
their gap.

It is also not the instrument for "can the training path fit a finite set at
all" -- gate 0c is, at 512 fixed instances with train and eval identical. Ad
hoc controls here duplicate a registered gate and should not be reported as
findings about the apparatus.

Two published literatures cover this task family:

  MQAR, in-context retrieval, the D=1 case. HazyResearch/zoology sweeps RWKV-7
  at this exact shape -- d_model in {64,128,256}, n_layers=2, head_dim=64 --
  over np.logspace(-3,-1.5,4) with max_epochs=32 and batch 256. Its claim is
  about model *dimension*: recurrent and gated-convolution models need d to
  scale with the MQAR input sequence length to stay accurate, where attention
  does not.

  That is a width bound, and this tool measures a data bound. They are not the
  same claim and Zoology does not license translating one into the other --
  note that what is observed here runs mildly against a pure width reading,
  since d is held at 128 while associations go 4 to 64 and the failure is
  rescued by more memories rather than more width. Treat the parallel as an
  open question, not as support.

  The S_n group word problem, the D>1 case. DeltaProduct uses lr 1e-3 with
  cosine annealing, 100 epochs, batch 1024, and reports S_n is solvable by one
  layer with n_h = n-1, 3 layers with n_h > 1, or 4 layers with n_h = 1. RWKV-7
  takes one delta step per token, so it is the n_h = 1 case: 4 layers.

This is an analysis tool, not a gate and not a CI test. Its output is a
scientific estimate; nothing here decides a pass or a fail.

Note on the memory count. Zoology trains MQAR on 20k-100k examples. A cell here
draws `queries_per_memory` queries from each memory, so the number of *distinct
memories* -- not the instance count -- is what determines whether memorising is
cheaper than retrieving. That is the axis --memories sweeps.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exp0.config import ModelConfig, TrainConfig
from exp0.train import create_model, set_seed
from exp1.dataset import PointerChaseDataset, exp1_collate_fn
from exp1.pointer_chase import ChaseSpec, generate_dataset
from exp1.train import evaluate_vway_accuracy, forward_logits
from rosa_compute.diagnostics import get_artifact_environment, get_git_commit

# Defined after the imports, not before: ruff's E402 allows a bare sys.path
# mutation ahead of them, but any other statement there makes every import
# below it an error.
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_bank(memories, *, depth, seed, spec, queries_per_memory,
               num_nodes, num_maps):
    return PointerChaseDataset(
        generate_dataset(memories, queries_per_memory=queries_per_memory,
                         depth=depth, seed=seed, num_nodes=num_nodes,
                         num_maps=num_maps),
        spec)


def run_cell(*, lr, d_model, layers, memories, num_nodes, num_maps, args,
             spec, val_ds, device):
    train_ds = build_bank(memories, depth=args.depth, seed=args.train_data_seed,
                          spec=spec, queries_per_memory=args.queries_per_memory,
                          num_nodes=num_nodes, num_maps=num_maps)
    # Scored on a slice of the training bank as well. The gap between the two is
    # the measurement: chance on both means it has not learned, high train with
    # chance held out means it has memorised, and only both rising together is
    # in-context retrieval.
    probe_n = min(len(train_ds), len(val_ds))
    probe_n -= probe_n % args.batch_size
    train_probe = PointerChaseDataset(train_ds.instances[:probe_n], spec)

    set_seed(args.model_seed)
    cfg = ModelConfig(architecture="rwkv", hidden_size=d_model,
                      num_hidden_layers=layers,
                      num_attention_heads=max(1, d_model // 64), head_dim=64,
                      vocab_size=num_nodes, rwkv_kernel=args.rwkv_kernel)
    model = create_model(cfg, d_input=spec.d_input,
                         compact_reduced_features=False).to(device)
    params = sum(p.numel() for p in model.parameters())
    if args.compile:
        model.backbone = torch.compile(model.backbone,
                                       backend=args.compile_backend)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    tcfg = TrainConfig(seed=args.model_seed, batch_size=args.batch_size,
                       precision=args.precision, epochs=args.epochs)
    gen = torch.Generator().manual_seed(args.shuffle_seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        generator=gen, drop_last=True,
                        collate_fn=exp1_collate_fn, num_workers=0,
                        pin_memory=True)

    accs, losses = [], []
    started = time.time()
    for epoch in range(args.epochs):
        if args.fresh_memories_per_epoch and epoch:
            # A new bank every epoch, so no memory is ever seen twice and
            # memorising is impossible by construction. Whatever accuracy
            # survives that is in-context retrieval, which is the quantity gate
            # 0a is meant to be measuring. Generation costs about 46 us per
            # instance -- 0.8 s for a 4,992-memory bank -- against an epoch of
            # roughly 108 s, so the effective bank is unbounded for ~1% more
            # wall clock and no extra memory.
            train_ds = build_bank(memories, depth=args.depth,
                                  seed=args.train_data_seed + epoch, spec=spec,
                                  queries_per_memory=args.queries_per_memory,
                                  num_nodes=num_nodes,
                                  num_maps=num_maps)
            loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, generator=gen, drop_last=True,
                                collate_fn=exp1_collate_fn, num_workers=0,
                                pin_memory=True)
        model.train()
        total = torch.zeros((), device=device, dtype=torch.float64)
        for batch in loader:
            x = batch["input_tuples"].to(device, non_blocking=True)
            y = batch["targets"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(forward_logits(model, x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += loss.detach().double()
        sched.step()
        losses.append((total / len(loader)).item())
        accs.append(evaluate_vway_accuracy(model, val_ds, tcfg, device, None))

    train_acc = evaluate_vway_accuracy(model, train_probe, tcfg, device, None)
    distinct = memories * (args.epochs if args.fresh_memories_per_epoch else 1)
    result = dict(lr=lr, d_model=d_model, layers=layers, memories=memories,
                  num_nodes=num_nodes, num_maps=num_maps,
                  associations=num_nodes * num_maps, chance=1.0 / num_nodes,
                  fresh_per_epoch=args.fresh_memories_per_epoch,
                  distinct_memories_seen=distinct,
                  params=params, instances=len(train_ds),
                  # Both are recorded, and `gap` and `z` below are derived from
                  # the final epoch, because that is the statistic the gates
                  # use. Deriving them from max(accs) silently builds a
                  # selection over 32-128 epochs into a number reported as if
                  # it were an outcome.
                  held_out_best=max(accs), held_out_final=accs[-1],
                  train_acc=train_acc, gap=train_acc - accs[-1],
                  final_loss=losses[-1], held_out_curve=accs,
                  train_losses=losses, seconds=time.time() - started)
    del model, opt
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if args.compile:
        torch._dynamo.reset()
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lrs", type=float, nargs="+", required=True)
    p.add_argument("--d-models", type=int, nargs="+", default=[128])
    p.add_argument("--layers", type=int, nargs="+", default=[2])
    p.add_argument("--memories", type=int, nargs="+", default=[4992],
                   help="distinct memories in the training bank; the axis that "
                        "decides whether memorising is cheaper than retrieving")
    p.add_argument("--val-memories", type=int, default=448)
    p.add_argument("--fresh-memories-per-epoch", action="store_true",
                   help="draw a new training bank every epoch, so no memory is "
                        "seen twice and memorising is impossible. The train "
                        "probe still scores epoch 0's bank, so a high train "
                        "accuracy under this flag means retention of a bank the "
                        "model has not seen for many epochs, not memorisation "
                        "of the current one")
    p.add_argument("--depth", type=int, default=1)
    # Swept, because the quantity the literature bounds is the number of
    # associations the recurrent state must hold, and that is num_nodes *
    # num_maps -- 64 at the gate's M=16, K=4. Chance is 1/num_nodes, so it
    # moves with the sweep and is reported per cell.
    p.add_argument("--num-nodes", type=int, nargs="+", default=[16])
    p.add_argument("--num-maps", type=int, nargs="+", default=[4])
    p.add_argument("--queries-per-memory", type=int, default=4)
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--precision", type=str, default="fp32",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--rwkv-kernel", type=str, default="reference",
                   choices=["reference", "cuda"])
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--model-seed", type=int, default=1001)
    p.add_argument("--train-data-seed", type=int, default=1004)
    p.add_argument("--val-data-seed", type=int, default=1005)
    p.add_argument("--shuffle-seed", type=int, default=4242)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-backend", type=str, default="cudagraphs",
                   choices=["inductor", "cudagraphs"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", type=str, default="",
                   help="free-text tag recorded in the artifact, e.g. the node")
    args = p.parse_args(argv)

    # Same alignment rule the gate runner enforces: cudagraphs replays into
    # fixed-size buffers, so a ragged final batch is fatal rather than slow.
    step = args.batch_size // math.gcd(args.batch_size, args.queries_per_memory)
    ragged = [f"{n} memories ({n * args.queries_per_memory} instances)"
              for n in list(args.memories) + [args.val_memories]
              if (n * args.queries_per_memory) % args.batch_size]
    if ragged and args.compile and args.compile_backend == "cudagraphs":
        p.error(f"cudagraphs needs every batch the same shape, but {', '.join(ragged)} "
                f"are not multiples of batch {args.batch_size}. Memory counts "
                f"must be multiples of {step}.")

    device = torch.device(args.device)
    print(f"depth={args.depth}   chance is 1/num_nodes and moves with the sweep")
    print(f"{'M':>3} {'K':>3} {'assoc':>6} {'chance':>7} {'lr':>9} {'d':>5} "
          f"{'L':>3} {'memories':>9} {'held/fin':>9} {'held/max':>9} "
          f"{'train':>8} {'gap':>8} {'z':>6} {'loss':>8} {'s':>6}")

    results = []
    for num_nodes in args.num_nodes:
        for num_maps in args.num_maps:
            # Rebuilt per (M, K): the field layout, and so d_input and the
            # learned input projection, depend on both. That is a real confound
            # on this axis -- a larger M is a wider input as well as more
            # associations -- which is why params is recorded per cell.
            spec = ChaseSpec(num_nodes=num_nodes, num_maps=num_maps,
                             max_depth=max(32, args.depth))
            val_ds = build_bank(args.val_memories, depth=args.depth,
                                seed=args.val_data_seed, spec=spec,
                                queries_per_memory=args.queries_per_memory,
                                num_nodes=num_nodes, num_maps=num_maps)
            chance = 1.0 / num_nodes
            se = math.sqrt(chance * (1 - chance) / args.val_memories)
            for memories in args.memories:
                for d_model in args.d_models:
                    for layers in args.layers:
                        for lr in args.lrs:
                            r = run_cell(lr=lr, d_model=d_model, layers=layers,
                                         memories=memories,
                                         num_nodes=num_nodes,
                                         num_maps=num_maps, args=args,
                                         spec=spec, val_ds=val_ds,
                                         device=device)
                            r["z"] = (r["held_out_final"] - chance) / se
                            results.append(r)
                            print(f"{num_nodes:>3} {num_maps:>3} "
                                  f"{num_nodes * num_maps:>6} {chance:>7.4f} "
                                  f"{lr:>9.1e} {d_model:>5} {layers:>3} "
                                  f"{memories:>9} {r['held_out_final']:>9.4f} "
                                  f"{r['held_out_best']:>9.4f} "
                                  f"{r['train_acc']:>8.4f} {r['gap']:>8.4f} "
                                  f"{r['z']:>+6.2f} {r['final_loss']:>8.4f} "
                                  f"{r['seconds']:>6.0f}")
                            sys.stdout.flush()

    # Recorded because these artifacts are read across the fleet and the cells
    # disagree between nodes. Without the compute capability and the torch and
    # CUDA versions, a cross-node comparison cannot separate architecture from
    # environment, and a reviewer holding only the JSON cannot audit either.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"label": args.label, "config": vars(args) | {"out": str(args.out)},
         "environment": get_artifact_environment(),
         "repo_commit": get_git_commit(str(REPO_ROOT)),
         "results": results}, indent=2, default=str))
    print(f"\nwrote {args.out}")

    # Clustered at the memory level. queries_per_memory queries share one
    # memory, so treating instances as independent understates the standard
    # error by sqrt(queries_per_memory) and turns noise into apparent signal.
    # That error has been made on this experiment once already and corrected;
    # the naive interval would report a 1.5-SE bump as a result.
    print(f"z is clustered at the memory level: n={args.val_memories} memories, "
          f"not {args.val_memories * args.queries_per_memory} queries.")
    above = [r for r in results if r["z"] >= 2]
    if above:
        print(f"{len(above)} of {len(results)} cells exceed chance at 2 SE:")
        for r in sorted(above, key=lambda r: -r["z"]):
            print(f"  M={r['num_nodes']} K={r['num_maps']} "
                  f"({r['associations']} associations) lr={r['lr']:.1e} "
                  f"d={r['d_model']} L={r['layers']} -> "
                  f"{r['held_out_final']:.4f} ({r['z']:+.2f} SE), "
                  f"train {r['train_acc']:.4f}")
    else:
        best = max(results, key=lambda r: r["z"])
        print(f"NO cell exceeded chance at 2 SE. The largest is {best['z']:+.2f} "
              f"SE, which is selection over noise rather than a result.")

    memorised = [r for r in results if r["gap"] > 0.2]
    if memorised:
        print(f"{len(memorised)} of {len(results)} cells memorised "
              f"(train - held-out > 0.2); largest gap "
              f"{max(r['gap'] for r in memorised):.4f}. A cell that memorises "
              f"is training fine and learning the wrong thing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
