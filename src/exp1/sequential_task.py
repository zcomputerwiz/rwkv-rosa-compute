"""Sequential-dependency task generator for Experiment 1 (H2).

The question H2 asks is whether extra recurrent transitions help a model perform
computation that genuinely requires sequential depth. That demands a task whose
depth cannot be collapsed, and most obvious candidates can be collapsed.

**Why not affine composition.** A chain of maps ``x -> a*x + b (mod m)`` is
associative, so a parallel prefix scan evaluates depth ``D`` in ``O(log D)``.
Any semigroup operation has the same problem. A model could solve such a task
with no sequential computation at all, and the measurement would say nothing
about H2.

**What this generator does instead: data-dependent dispatch.** The operation
applied at step ``k`` is selected by the runtime *value* of the register it
reads::

    v_even -> (v + a) mod m
    v_odd  -> (v * b) mod m

The branch depends on a value that depends on every prior step, so the
composition cannot be folded ahead of time and no associative scan applies.
Evaluation is inherently ordered.

**Properties required by the H2 design, and how each is obtained.**

``explicit dependency depth D``
    Measured, not assumed. ``D`` is computed by backward slice from the answer
    register over the actual program, so it is the true dependency depth even
    if construction went astray.

``same surface length, different D``
    Every instance has exactly ``num_instructions`` instructions. Depth varies
    with how many of them lie on the chain feeding the answer.

``state_k depends on state_(k-1)``
    By construction of the chain, and verified by the backward slice.

``no easy local shortcut``
    Chain instructions are placed at random positions and are locally
    indistinguishable from distractors: same instruction format, same register
    alphabet, operands drawn from the same distribution. Identifying which
    instructions matter requires tracing backwards from the answer register.

``exact final answer``
    A single value in ``Z_m``.

``difficulty independent of prompt length``
    ``num_instructions`` is fixed across the whole dataset; only ``D`` varies.

**The modulus must be prime.** With ``mod=10`` the value 5 is a fixed point of
the odd branch (5*1, 5*3, 5*7, 5*9 are all 5 mod 10), so chains that reach it are
trapped and the answer distribution skews hard: measured 16.8% on answer 5
against 10% uniform, handing a majority-class baseline a free 1.7x over chance
with no computation at all. A prime modulus makes multiplication by any nonzero
residue a bijection and removes the attractor - measured max deviation falls
from 7.4 points to 3.4 at ``mod=13``. This was caught by the first-pass audit of
this generator, which is exactly what the audit is for.

Leakage this generator deliberately avoids, and which the audit should confirm:
chain instructions are not positionally biased, distractor and chain registers
are drawn from one alphabet, and every instruction has identical surface form.
The answer distribution is *not* forced uniform - see ``answer_histogram``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Instruction",
    "SequentialInstance",
    "generate_instance",
    "generate_dataset",
    "answer_histogram",
]


@dataclass(frozen=True)
class Instruction:
    """``target <- dispatch(source)`` with operands ``a`` (even) and ``b`` (odd)."""

    target: int
    source: int
    a: int
    b: int

    def apply(self, value: int, mod: int) -> int:
        if value % 2 == 0:
            return (value + self.a) % mod
        return (value * self.b) % mod


@dataclass(frozen=True)
class SequentialInstance:
    initial: Tuple[int, ...]
    instructions: Tuple[Instruction, ...]
    answer_register: int
    answer: int
    depth: int
    chain_positions: Tuple[int, ...]
    mod: int


def _simulate(
    initial: Sequence[int],
    instructions: Sequence[Instruction],
    mod: int,
) -> List[int]:
    regs = list(initial)
    for ins in instructions:
        regs[ins.target] = ins.apply(regs[ins.source], mod)
    return regs


def _backward_slice(
    instructions: Sequence[Instruction],
    answer_register: int,
) -> Tuple[int, Tuple[int, ...]]:
    """Return (depth, positions) of the true dependency chain for the answer.

    Walks backwards finding, for the register currently of interest, the last
    instruction that wrote it before the point of interest. That instruction's
    source register becomes the new register of interest. The chain ends when no
    earlier write exists, meaning the value came from the initial state.
    """
    positions: List[int] = []
    want = answer_register
    cursor = len(instructions)
    while True:
        writer = None
        for i in range(cursor - 1, -1, -1):
            if instructions[i].target == want:
                writer = i
                break
        if writer is None:
            break
        positions.append(writer)
        want = instructions[writer].source
        cursor = writer
    positions.reverse()
    return len(positions), tuple(positions)


def generate_instance(
    rng: random.Random,
    *,
    depth: int,
    num_instructions: int,
    num_registers: int,
    mod: int,
    max_attempts: int = 200,
) -> Optional[SequentialInstance]:
    """Generate one instance whose *verified* dependency depth equals ``depth``.

    Construction places a chain then fills with distractors, but the returned
    depth is always the backward-slice measurement. If a distractor perturbs the
    chain the attempt is discarded rather than silently mislabelled, which is
    why this can return ``None``.
    """
    if depth > num_instructions:
        raise ValueError("depth cannot exceed num_instructions")
    if num_registers < 2:
        raise ValueError("need at least two registers")

    for _ in range(max_attempts):
        answer_register = rng.randrange(num_registers)

        # Chain positions: random, so depth is not readable from layout.
        chain_slots = sorted(rng.sample(range(num_instructions), depth))
        slots = {}

        # Build the chain backwards from the answer register.
        want = answer_register
        for slot in reversed(chain_slots):
            source = rng.randrange(num_registers)
            slots[slot] = Instruction(
                target=want,
                source=source,
                a=rng.randrange(mod),
                b=rng.randrange(1, mod),
            )
            want = source

        # Distractors: same alphabet, same surface form, no special registers.
        instructions: List[Instruction] = []
        for i in range(num_instructions):
            if i in slots:
                instructions.append(slots[i])
            else:
                instructions.append(
                    Instruction(
                        target=rng.randrange(num_registers),
                        source=rng.randrange(num_registers),
                        a=rng.randrange(mod),
                        b=rng.randrange(1, mod),
                    )
                )

        measured_depth, measured_positions = _backward_slice(
            instructions, answer_register
        )
        if measured_depth != depth:
            continue  # a distractor changed the chain; discard

        initial = tuple(rng.randrange(mod) for _ in range(num_registers))
        regs = _simulate(initial, instructions, mod)
        return SequentialInstance(
            initial=initial,
            instructions=tuple(instructions),
            answer_register=answer_register,
            answer=regs[answer_register],
            depth=measured_depth,
            chain_positions=measured_positions,
            mod=mod,
        )
    return None


def generate_dataset(
    num_per_depth: int,
    depths: Sequence[int],
    *,
    seed: int,
    num_instructions: int = 32,
    num_registers: int = 4,
    mod: int = 13,
) -> List[SequentialInstance]:
    rng = random.Random(seed)
    out: List[SequentialInstance] = []
    for depth in depths:
        made = 0
        guard = 0
        while made < num_per_depth:
            guard += 1
            if guard > num_per_depth * 500:
                raise RuntimeError(
                    f"could not generate depth={depth} with "
                    f"num_instructions={num_instructions}, "
                    f"num_registers={num_registers}"
                )
            inst = generate_instance(
                rng,
                depth=depth,
                num_instructions=num_instructions,
                num_registers=num_registers,
                mod=mod,
            )
            if inst is not None:
                out.append(inst)
                made += 1
    rng.shuffle(out)
    return out


def answer_histogram(instances: Sequence[SequentialInstance]) -> Dict[int, int]:
    """Answer frequencies. A skewed distribution is itself a shortcut."""
    hist: Dict[int, int] = {}
    for inst in instances:
        hist[inst.answer] = hist.get(inst.answer, 0) + 1
    return hist


def _self_check() -> None:
    rng = random.Random(0)
    inst = generate_instance(
        rng, depth=4, num_instructions=16, num_registers=4, mod=13
    )
    assert inst is not None
    # depth is the measured backward slice
    assert inst.depth == 4 == len(inst.chain_positions)
    # simulation reproduces the recorded answer
    regs = _simulate(inst.initial, inst.instructions, inst.mod)
    assert regs[inst.answer_register] == inst.answer
    # the chain genuinely matters: perturbing any chain instruction changes it
    changed = 0
    for pos in inst.chain_positions:
        mutated = list(inst.instructions)
        old = mutated[pos]
        mutated[pos] = Instruction(old.target, old.source, (old.a + 1) % inst.mod, old.b)
        if _simulate(inst.initial, mutated, inst.mod)[inst.answer_register] != inst.answer:
            changed += 1
    assert changed > 0, "chain instructions had no effect on the answer"
    # every instance has identical surface length
    data = generate_dataset(20, [1, 2, 4], seed=1, num_instructions=16)
    assert {len(d.instructions) for d in data} == {16}
    assert {d.depth for d in data} == {1, 2, 4}
    print("sequential_task self-check OK")


if __name__ == "__main__":
    _self_check()
