"""Sequence encoding for the Experiment 1 sequential task.

Mirrors the Experiment 0 convention: a instance becomes a ``[positions,
d_input]`` multi-hot float32 tensor, one row per sequence position, so the same
training harness consumes it.

Layout::

    [ init_0 .. init_{R-1} ] [ ins_0 .. ins_{L-1} ] [ silent_0 .. silent_{N-1} ] [ query ]

**All three H2 arms are expressed by one parameter**, ``silent_kind``, so the
neutral-token control never requires a rewrite:

``silent_kind=None``   arm A, N=0 - no silent positions at all
``silent_kind="scratchpad"``  arm B - N silent positions, own kind slot
``silent_kind="neutral"``     arm C - N silent positions, different kind slot

Arms B and C are byte-identical in structure and differ only in which kind bit
is set, which is what makes ``B > C`` interpretable as scratchpad-specific
representation rather than an artefact of sequence shape. If the two arms
differed in length or layout, that comparison would be confounded.

The answer is read at the final ``query`` position, and is one of ``mod``
classes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from exp1.sequential_task import SequentialInstance

__all__ = ["EncodingSpec", "encode_instance", "encode_batch"]

_KINDS = ("init", "instruction", "scratchpad", "neutral", "query")
_KIND_INDEX = {name: i for i, name in enumerate(_KINDS)}


class EncodingSpec:
    """Field layout for one sequence position.

    Every position carries every field slot; unused slots stay zero. That keeps
    positions homogeneous, so an instruction is not distinguishable from a
    silent position by vector *shape* - only by which bits are set.
    """

    def __init__(self, *, num_registers: int, mod: int, num_instructions: int,
                 num_silent: int) -> None:
        if num_silent < 0:
            raise ValueError("num_silent must be non-negative")
        self.num_registers = num_registers
        self.mod = mod
        self.num_instructions = num_instructions
        self.num_silent = num_silent

        self.seq_len = num_registers + num_instructions + num_silent + 1

        # field offsets
        self.off_kind = 0
        self.off_target = self.off_kind + len(_KINDS)
        self.off_source = self.off_target + num_registers
        self.off_a = self.off_source + num_registers
        self.off_b = self.off_a + mod
        self.off_c = self.off_b + mod
        self.off_value = self.off_c + mod
        self.off_position = self.off_value + mod
        self.d_input = self.off_position + self.seq_len

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"EncodingSpec(seq_len={self.seq_len}, d_input={self.d_input}, "
                f"R={self.num_registers}, mod={self.mod}, "
                f"L={self.num_instructions}, N={self.num_silent})")


def encode_instance(
    instance: SequentialInstance,
    spec: EncodingSpec,
    *,
    silent_kind: Optional[str] = None,
) -> torch.Tensor:
    """Encode one instance to ``[seq_len, d_input]`` float32."""
    if silent_kind not in (None, "scratchpad", "neutral"):
        raise ValueError("silent_kind must be None, 'scratchpad' or 'neutral'")
    if spec.num_silent and silent_kind is None:
        raise ValueError("num_silent > 0 requires a silent_kind")
    if silent_kind is not None and spec.num_silent == 0:
        raise ValueError("silent_kind set but num_silent is 0")

    x = torch.zeros((spec.seq_len, spec.d_input), dtype=torch.float32)
    pos = 0

    def mark(kind: str) -> int:
        nonlocal pos
        x[pos, spec.off_kind + _KIND_INDEX[kind]] = 1.0
        x[pos, spec.off_position + pos] = 1.0
        here = pos
        pos += 1
        return here

    # initial register values: register identity in the target slot, value in value
    for reg, value in enumerate(instance.initial):
        i = mark("init")
        x[i, spec.off_target + reg] = 1.0
        x[i, spec.off_value + value] = 1.0

    for ins in instance.instructions:
        i = mark("instruction")
        x[i, spec.off_target + ins.target] = 1.0
        x[i, spec.off_source + ins.source] = 1.0
        x[i, spec.off_a + ins.a] = 1.0
        x[i, spec.off_b + ins.b] = 1.0
        x[i, spec.off_c + ins.c] = 1.0

    for _ in range(spec.num_silent):
        mark(silent_kind)  # kind bit is the ONLY difference between arms B and C

    # query names the register to read; carries no value
    i = mark("query")
    x[i, spec.off_target + instance.answer_register] = 1.0

    assert pos == spec.seq_len, (pos, spec.seq_len)
    return x


def encode_batch(
    instances: Sequence[SequentialInstance],
    spec: EncodingSpec,
    *,
    silent_kind: Optional[str] = None,
) -> tuple:
    """Return ``(inputs [B, seq_len, d_input], answers [B], depths [B])``."""
    xs: List[torch.Tensor] = []
    for inst in instances:
        xs.append(encode_instance(inst, spec, silent_kind=silent_kind))
    inputs = torch.stack(xs)
    answers = torch.tensor([i.answer for i in instances], dtype=torch.long)
    depths = torch.tensor([i.depth for i in instances], dtype=torch.long)
    return inputs, answers, depths


def _self_check() -> None:
    import random

    from exp1.sequential_task import generate_instance

    rng = random.Random(0)
    inst = generate_instance(rng, depth=3, num_instructions=8,
                             num_registers=4, mod=13)
    assert inst is not None

    # Arm A: no silent positions
    spec_a = EncodingSpec(num_registers=4, mod=13, num_instructions=8, num_silent=0)
    a = encode_instance(inst, spec_a)
    assert a.shape == (4 + 8 + 0 + 1, spec_a.d_input)

    # Arms B and C: identical shape, differing in exactly one column block
    spec_bc = EncodingSpec(num_registers=4, mod=13, num_instructions=8, num_silent=5)
    b = encode_instance(inst, spec_bc, silent_kind="scratchpad")
    c = encode_instance(inst, spec_bc, silent_kind="neutral")
    assert b.shape == c.shape == (4 + 8 + 5 + 1, spec_bc.d_input)
    diff = (b != c).any(dim=1)
    assert int(diff.sum()) == 5, "arms B and C must differ at exactly the silent rows"
    kind_cols = slice(spec_bc.off_kind, spec_bc.off_kind + len(_KINDS))
    assert torch.equal(b[:, spec_bc.off_target:], c[:, spec_bc.off_target:]), \
        "arms B and C must differ only in the kind field"
    assert not torch.equal(b[:, kind_cols], c[:, kind_cols])

    # every position has exactly one kind bit and one position bit
    for enc, sp in ((a, spec_a), (b, spec_bc)):
        kinds = enc[:, sp.off_kind:sp.off_kind + len(_KINDS)].sum(dim=1)
        assert torch.all(kinds == 1.0)
        posn = enc[:, sp.off_position:].sum(dim=1)
        assert torch.all(posn == 1.0)

    # N changes sequence length only through silent positions
    assert spec_bc.seq_len - spec_a.seq_len == 5

    inputs, answers, depths = encode_batch([inst, inst], spec_a)
    assert inputs.shape[0] == 2 and answers.shape == (2,) and depths.shape == (2,)
    print("encoding self-check OK", spec_bc)


if __name__ == "__main__":
    _self_check()
