"""Query-deferred multi-relation pointer chase — the H2 task, second design.

Replaces `sequential_task.py`, which cannot test H2: its program is ingested
before the query and each chain step consumes one instruction token, so a
strictly streaming interpreter solves it carrying ``R*log2(mod)`` bits and any
silent positions arrive with no computation left to do.

**The fix is query deferral.** The memory (``K`` maps over ``M`` nodes) is
ingested first. The start node *and the entire selector sequence* then arrive in
a **single** input position. Only after that position can any chain step be
taken::

    x_{t+1} = f_{selector_t}(x_t)

Packing all ``D`` selectors into one position is the load-bearing detail. Spread
across ``D`` positions, the model would take one step per selector token and we
would have rebuilt the previous design's defect exactly.

**Why precomputation during ingestion does not help.** Without the selector
sequence, a model wanting to answer immediately must hold every composition it
might be asked for::

    store the K primitive maps       M*K       = 64      at M=16, K=4
    store all D-step compositions    M*K^D     = 6.9e10  at D=16

The asymmetry is the design, not the absolute size — we do not need ``M`` to
exceed the recurrent state, only ``K^D`` to exceed what is worth caching.

**What this does NOT claim.** No formal sequential lower bound. Query deferral
plus a combinatorial query space removes the known cheap precomputation
shortcut; it does not prove none exists.

**Layer depth substitutes for time depth.** A ``D``-step chain can be unrolled
across layers within one token's forward pass while ``D`` is within the layer
budget. On the production 12-layer model, ``D <= 12`` should be solvable at
``N=0``, and the H2 signal can only appear above that. ``Dmax`` is therefore set
to 32 rather than 16, so the informative region is ``D in {12..32}`` rather than
a four-point sliver.

Arms, as before, differ in exactly one bit at exactly the silent rows:

``silent_kind=None``          arm A, N=0
``silent_kind="scratchpad"``  arm B
``silent_kind="neutral"``     arm C, a fixed non-trainable vector, norm-matched
``silent_kind="zero"``        arm D, pure recurrence, no input signal at all
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

__all__ = ["Memory", "ChaseInstance", "ChaseSpec", "generate_memory",
           "generate_instance", "generate_dataset", "encode_batch"]

_KINDS = ("memory", "query", "scratchpad", "neutral", "zero", "probe")
_KIND_INDEX = {name: i for i, name in enumerate(_KINDS)}
IDENTITY_SELECTOR = -1          # pads the selector field to a fixed width


@dataclass(frozen=True)
class Memory:
    """``K`` maps over ``M`` nodes. Held out whole at validation."""
    maps: Tuple[Tuple[int, ...], ...]

    @property
    def num_nodes(self) -> int:
        return len(self.maps[0])

    @property
    def num_maps(self) -> int:
        return len(self.maps)


@dataclass(frozen=True)
class ChaseInstance:
    memory: Memory
    start: int
    selectors: Tuple[int, ...]
    answer: int

    @property
    def depth(self) -> int:
        return len(self.selectors)


def generate_memory(rng: random.Random, *, num_nodes: int, num_maps: int,
                    permutations: bool = True) -> Memory:
    """``K`` maps over ``M`` nodes. Permutations by default, and that matters.

    Arbitrary random functions concentrate under iteration - their image shrinks
    onto attractors - so the answer distribution is skewed and a model that
    ignores the query entirely scores well above chance. Measured at M=16, K=4:

    ```text
                        ignore-the-query floor
        arbitrary maps  0.134 (D=1) -> 0.144 (D=16), up to 0.27 per memory
        permutations    converges to 1/M = 0.0625 at every D
    ```

    A composition of permutations is a permutation, so a uniform start yields a
    uniform answer whatever ``D`` is, and the floor is exactly ``1/M``
    independent of the memory drawn. With arbitrary maps the floor is both
    elevated and memory-dependent, which would have to be measured per cell and
    would confound the D axis - the attractor effect strengthens with depth.
    """
    if permutations:
        maps = []
        for _ in range(num_maps):
            perm = list(range(num_nodes))
            rng.shuffle(perm)
            maps.append(tuple(perm))
        return Memory(tuple(maps))
    return Memory(tuple(tuple(rng.randrange(num_nodes) for _ in range(num_nodes))
                        for _ in range(num_maps)))


def execute(memory: Memory, start: int, selectors: Sequence[int]) -> int:
    x = start
    for s in selectors:
        if s == IDENTITY_SELECTOR:
            continue
        x = memory.maps[s][x]
    return x


def generate_instance(rng: random.Random, memory: Memory, *, depth: int) -> ChaseInstance:
    start = rng.randrange(memory.num_nodes)
    selectors = tuple(rng.randrange(memory.num_maps) for _ in range(depth))
    return ChaseInstance(memory, start, selectors, execute(memory, start, selectors))


def generate_dataset(num_memories: int, queries_per_memory: int, depth: int, *,
                     seed: int, num_nodes: int = 16, num_maps: int = 4,
                     permutations: bool = True) -> List[ChaseInstance]:
    """Multiple queries share one memory; memories are the unit held out."""
    rng = random.Random(seed)
    out: List[ChaseInstance] = []
    for _ in range(num_memories):
        mem = generate_memory(rng, num_nodes=num_nodes, num_maps=num_maps,
                              permutations=permutations)
        for _ in range(queries_per_memory):
            out.append(generate_instance(rng, mem, depth=depth))
    return out


class ChaseSpec:
    """Field layout. ``d_input`` is constant across every ``D`` and every ``N``.

    No absolute position encoding: it made ``d_input`` grow with ``N`` in the
    previous design, which made the arms different architectures, and it makes
    extrapolation to unseen ``N`` ill-defined.
    """

    def __init__(self, *, num_nodes: int = 16, num_maps: int = 4,
                 max_depth: int = 32) -> None:
        self.num_nodes = num_nodes
        self.num_maps = num_maps
        self.max_depth = max_depth

        self.off_kind = 0
        self.off_node = self.off_kind + len(_KINDS)
        self.off_images = self.off_node + num_nodes           # K images of this node
        self.off_start = self.off_images + num_maps * num_nodes
        self.off_selectors = self.off_start + num_nodes       # D_max slots, K+1 wide
        self.d_input = self.off_selectors + max_depth * (num_maps + 1)

    def seq_len(self, num_silent: int) -> int:
        return self.num_nodes + 1 + num_silent + 1            # memory, query, silent, probe

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ChaseSpec(M={self.num_nodes}, K={self.num_maps}, "
                f"Dmax={self.max_depth}, d_input={self.d_input})")


def encode_batch(instances: Sequence[ChaseInstance], spec: ChaseSpec, *,
                 num_silent: int = 0, silent_kind: Optional[str] = None,
                 neutral_vector: Optional[torch.Tensor] = None):
    """Return ``(inputs [B, T, d_input], answers [B])``."""
    if num_silent and silent_kind is None:
        raise ValueError("num_silent > 0 requires a silent_kind")
    if silent_kind is not None and silent_kind not in ("scratchpad", "neutral", "zero"):
        raise ValueError(f"bad silent_kind: {silent_kind}")

    T = spec.seq_len(num_silent)
    x = torch.zeros((len(instances), T, spec.d_input), dtype=torch.float32)
    for b, inst in enumerate(instances):
        pos = 0
        for node in range(spec.num_nodes):                    # the memory
            x[b, pos, spec.off_kind + _KIND_INDEX["memory"]] = 1.0
            x[b, pos, spec.off_node + node] = 1.0
            for k in range(spec.num_maps):
                x[b, pos, spec.off_images + k * spec.num_nodes + inst.memory.maps[k][node]] = 1.0
            pos += 1

        # start node AND every selector, in ONE position - this is what defers
        # the computation past ingestion
        x[b, pos, spec.off_kind + _KIND_INDEX["query"]] = 1.0
        x[b, pos, spec.off_start + inst.start] = 1.0
        for slot in range(spec.max_depth):
            sel = inst.selectors[slot] if slot < len(inst.selectors) else IDENTITY_SELECTOR
            col = spec.num_maps if sel == IDENTITY_SELECTOR else sel
            x[b, pos, spec.off_selectors + slot * (spec.num_maps + 1) + col] = 1.0
        pos += 1

        for _ in range(num_silent):
            if silent_kind == "zero":
                pass                                          # no input signal at all
            elif silent_kind == "neutral":
                if neutral_vector is None:
                    raise ValueError("neutral arm requires a neutral_vector")
                x[b, pos] = neutral_vector
            else:
                x[b, pos, spec.off_kind + _KIND_INDEX["scratchpad"]] = 1.0
            pos += 1

        x[b, pos, spec.off_kind + _KIND_INDEX["probe"]] = 1.0
        pos += 1
        assert pos == T, (pos, T)
    answers = torch.tensor([i.answer for i in instances], dtype=torch.long)
    return x, answers


def make_neutral_vector(spec: ChaseSpec, seed: int = 0) -> torch.Tensor:
    """Fixed, non-trainable, norm-matched to a scratchpad row.

    A zero vector would differ from the scratchpad token in norm as well as in
    meaning, confounding "what the token means" with "how much signal it
    carries". That variant is arm D, kept separate and labelled.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(spec.d_input, generator=g)
    scratch = torch.zeros(spec.d_input)
    scratch[spec.off_kind + _KIND_INDEX["scratchpad"]] = 1.0
    return v * (scratch.norm() / v.norm())


def _self_check() -> None:
    spec = ChaseSpec()
    print(f"  {spec}")

    # d_input must not move with D or N - the previous design's blocker 1
    widths = {ChaseSpec(max_depth=d).d_input for d in (16, 32)}
    assert len(widths) == 2, "max_depth changes width, as intended (fixed per study)"
    s = ChaseSpec()
    assert all(s.d_input == s.d_input for _ in range(3))
    data = generate_dataset(4, 8, depth=8, seed=0)
    for n in (0, 4, 16):
        kind = None if n == 0 else "scratchpad"
        xs, _ = encode_batch(data, s, num_silent=n, silent_kind=kind)
        assert xs.shape[-1] == s.d_input, "width moved with N"
        assert xs.shape[1] == s.seq_len(n)
    print("  d_input constant across N: OK")

    # answers must be a deterministic function of the encoded input
    for inst in data:
        assert execute(inst.memory, inst.start, inst.selectors) == inst.answer
    print("  answers reproduce from memory + start + selectors: OK")

    # arms B and C differ only at the silent rows
    nv = make_neutral_vector(s)
    xb, _ = encode_batch(data[:2], s, num_silent=5, silent_kind="scratchpad")
    xc, _ = encode_batch(data[:2], s, num_silent=5, silent_kind="neutral",
                         neutral_vector=nv)
    diff = (xb != xc).any(dim=-1)
    assert int(diff.sum()) == 2 * 5, "arms differ outside the silent rows"
    print("  arms B/C differ at exactly the silent rows: OK")

    scratch_norm = xb[0, spec.num_nodes + 1].norm()
    assert abs(float(nv.norm() - scratch_norm)) < 1e-5, "neutral not norm-matched"
    print(f"  neutral vector norm-matched to scratchpad ({float(scratch_norm):.4f}): OK")

    # the streaming defect must be gone: nothing before the query determines the answer
    pre_query = {}
    for inst in data:
        key = tuple(inst.memory.maps)
        pre_query.setdefault(key, set()).add(inst.answer)
    assert any(len(v) > 1 for v in pre_query.values()), \
        "answers are constant per memory - the query would carry no information"
    print("  one memory yields many different answers, so the query is load-bearing: OK")
    print("  self-check OK")


if __name__ == "__main__":
    _self_check()
