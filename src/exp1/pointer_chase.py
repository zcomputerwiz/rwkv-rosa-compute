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

**That count is incomplete, and the gap matters.** It prices only the extreme
strategies. Caching intermediate *blocks* and chaining them is far cheaper::

    length-2 table   M*K^2 =   256 entries   D=32 becomes ~16 lookups
    length-3 table   M*K^3 = 1,024 entries   D=32 becomes ~11 lookups

The block table must be built per memory rather than memorised in weights, but
the apparatus permits exactly that: several queries share one memory, and the
recurrent state after ingestion is query-independent. Whether such a table is
actually learned is an empirical question, audited before gate 1; it is not
ruled out by counting.

**What this does NOT claim.** No formal sequential lower bound. Query deferral
plus a combinatorial query space removes the known cheap precomputation
shortcut; it does not prove none exists.

**``D`` is composition length, not certified serial neural depth.** A
``D``-step chain can be unrolled across layers within one token's forward pass
while ``D`` is within the layer budget, so larger ``D`` plausibly demands more
of the model. It does not follow that ``D`` steps are *required*: a per-memory
block table reduces the query-time chain to roughly ``D/b`` steps, and nothing
here rules that out. ``Dmax`` is 32 rather than 16 to give the depth axis
range. The region above the layer count is where a serial-depth effect would
be easiest to see, not a region where serial depth is the only available
explanation.

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

import numpy as np
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

    **A second, higher floor exists for a model that reads the query.** The
    table above is the floor for ignoring the query entirely. Selectors are
    drawn independently, so a query can repeat one, and a repeated word need
    not induce a uniform composition.

    One sufficient condition is exact and worth stating: **if any selector
    appears exactly once, the composition is exactly uniform.** Condition on
    every other map and the word becomes ``A pi_j B`` for fixed ``A``, ``B``;
    ``pi -> A pi B`` is a bijection of ``S_M``, so the product is uniform. It
    is sufficient, *not* necessary. At ``M=16, D=17`` the all-identical word
    ``pi^17`` is also exactly uniform, because ``gcd(17, exponent(S_16)) = 1``
    makes ``g -> g^17`` a bijection.

    For an all-identical word the expected fixed-point count is::

        E[fix(pi^D)] = #{ l <= M : l divides D }

    restricted to ``l <= M``, because a divisor larger than ``M`` cannot be a
    cycle length. At ``M=16, D=32`` that is 5, not the 6 an unrestricted
    divisor count gives.

    **The size of the resulting bias is not established here.** Pooled
    ``P(answer == start)`` averages over words and can sit at ``1/M`` while
    per-word deviations cancel. The quantity that bounds a word-aware model is
    ``E_w[max(p_w, (1 - p_w)/(M - 1))]``, and estimating it needs nested
    simulation with memory-level clustering - an analysis tool, not a
    docstring. The primary H2 estimand is a paired within-``D`` contrast
    across ``N``, which cancels any depth-specific floor common to the matched
    arms.
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


def generate_instance(rng: random.Random, memory: Memory, *, depth: int, max_depth: int = 64) -> ChaseInstance:
    start = rng.randrange(memory.num_nodes)
    if depth > max_depth:
        raise ValueError(f"depth {depth} exceeds max_depth {max_depth}")
    all_selectors = tuple(rng.randrange(memory.num_maps) for _ in range(max_depth))
    selectors = all_selectors[:depth]
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
    """Return ``(inputs [B, T, d_input], answers [B])``.

    Written as scatters rather than as a loop of scalar assignments. The loop
    form cost one Python-level ``__setitem__`` per set bit -- ``M*(2+K) + 1 +
    Dmax + N + 1`` of them per instance, or 2.6 million for a 19,968-instance
    training bank at ``M=16, K=4, Dmax=32``, and 15.5 s of a run whose training
    phase is 33 s once the backbone is compiled.

    Every write is ``= 1.0`` into a zeroed tensor, so the operation is a set
    union rather than an accumulation: repeated indices are idempotent and the
    scatter reproduces the loop bit for bit. ``encoding_matches_reference`` in
    the test module asserts exactly that across the arms and the padding cases,
    because "faster and equal" is the only version of this change worth having.
    """
    if num_silent and silent_kind is None:
        raise ValueError("num_silent > 0 requires a silent_kind")
    if silent_kind is not None and silent_kind not in ("scratchpad", "neutral", "zero"):
        raise ValueError(f"bad silent_kind: {silent_kind}")
    B = len(instances)
    M, K, Dmax = spec.num_nodes, spec.num_maps, spec.max_depth
    T = spec.seq_len(num_silent)
    x = torch.zeros((B, T, spec.d_input), dtype=torch.float32)
    if B == 0:
        return x, torch.zeros((0,), dtype=torch.long)

    # Checked here rather than at the top of the function: the original raised
    # from inside the per-row loop, so an empty batch never reached it and
    # returned empty tensors even with no neutral vector. Hoisting the guard
    # above the early return would turn that into a ValueError.
    if silent_kind == "neutral" and num_silent and neutral_vector is None:
        raise ValueError("neutral arm requires a neutral_vector")

    node = torch.arange(M)
    query_pos = M
    probe_pos = T - 1

    # --- memory rows, positions 0..M-1 ---------------------------------------
    x[:, :M, spec.off_kind + _KIND_INDEX["memory"]] = 1.0
    x[:, node, spec.off_node + node] = 1.0                     # node identity

    # maps[b, k, n] is the image of node n under map k. The column for that
    # image is off_images + k*M + image, and the row is n.
    # Through numpy: building a [B, K, M] tensor from nested Python tuples is
    # markedly slower via torch.tensor than via np.array plus from_numpy.
    maps = torch.from_numpy(
        np.array([inst.memory.maps for inst in instances], dtype=np.int64))
    image_col = spec.off_images + (torch.arange(K).view(1, K, 1) * M) + maps
    x[torch.arange(B).view(B, 1, 1), node.view(1, 1, M), image_col] = 1.0

    # --- query row: the start node and the whole selector word, one position --
    x[:, query_pos, spec.off_kind + _KIND_INDEX["query"]] = 1.0
    starts = torch.tensor([inst.start for inst in instances], dtype=torch.long)
    x[torch.arange(B), query_pos, spec.off_start + starts] = 1.0

    # Slots past the instance's depth hold IDENTITY_SELECTOR, which occupies
    # column K of its slot -- so K is the fill value, not a sentinel to replace
    # later. Selectors beyond Dmax are dropped, matching the loop's fixed range.
    words = [inst.selectors[:Dmax] for inst in instances]
    sel_np = np.full((B, Dmax), K, dtype=np.int64)
    widths = {len(w) for w in words}
    if len(widths) == 1:
        # Every instance in a cell shares one depth, so the whole block lands in
        # a single array construction. The ragged branch below is the general
        # case and stays correct, it is just not the one that runs.
        width = widths.pop()
        if width:
            sel_np[:, :width] = np.array(words, dtype=np.int64)
    else:
        for b, word in enumerate(words):
            if word:
                sel_np[b, :len(word)] = word
    sel = torch.from_numpy(sel_np)
    sel[sel == IDENTITY_SELECTOR] = K
    slot_col = spec.off_selectors + torch.arange(Dmax) * (K + 1) + sel
    x[torch.arange(B).view(B, 1), query_pos, slot_col] = 1.0

    # --- silent rows, then the probe ------------------------------------------
    if num_silent:
        rows = slice(query_pos + 1, query_pos + 1 + num_silent)
        if silent_kind == "neutral":
            x[:, rows, :] = neutral_vector
        elif silent_kind == "scratchpad":
            x[:, rows, spec.off_kind + _KIND_INDEX["scratchpad"]] = 1.0
        # "zero" writes nothing at all: no input signal, which is the arm

    x[:, probe_pos, spec.off_kind + _KIND_INDEX["probe"]] = 1.0

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
