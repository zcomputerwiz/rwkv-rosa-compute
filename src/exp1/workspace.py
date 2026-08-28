"""Latent workspace: ``M`` addressable slots refined for ``K`` steps.

The experiment this exists for is a 2x2 over ``(M, K)``. That design only
measures what it claims to measure if moving along either axis changes *what
the model can do* and not *how many parameters it has*. An earlier version of
this line of work failed exactly there: the two arms carried different
parameter counts, so the comparison measured capacity rather than mechanism.

Two invariants therefore hold by construction, and are asserted by the tests:

``M`` costs no parameters
    Slot ``i`` is seeded as ``Z_i = S + c_i`` where the ``c_i`` are fixed,
    non-trainable, mutually orthonormal offsets, taken as the *first* ``M`` rows
    of one table sized at ``m_max``. Slot 0 of ``M=1`` is bit-identical to slot
    0 of ``M=8``: the small arm is a strict prefix of the large one, not a
    differently-seeded object.

``K`` costs no parameters
    The refinement block is weight-tied across steps. Running it eight times is
    the same weights applied eight times, so ``K`` buys serial depth and
    nothing else. This is the whole hypothesis: if extra steps help, they help
    because of iteration, not because of capacity.

Cross-slot communication is **learned content-based routing** with shared
projections. An earlier revision used a parameter-free mean and justified it on
the grounds that learned routing would make parameter count scale with ``M``.
That justification was wrong, and a reviewer was right to reject it: attention
projections are ``d_model x d_model`` regardless of how many slots attend
through them, so routing can be learned while the invariant still holds
exactly. A global mean is also a genuine information bottleneck, and a screen
built on it risks a negative result that says more about the coupling than
about the hypothesis.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def canonical_offsets(m_max: int, d_model: int) -> torch.Tensor:
    """``m_max x d_model`` orthonormal rows, identical on every machine.

    Built by exact construction rather than by decomposition. The previous
    version used ``torch.linalg.qr`` on seeded noise, which is LAPACK-backed:
    pinning the sign against ``diag(R)`` removes the sign ambiguity but not the
    floating-point differences between LAPACK builds, so two nodes could hold
    tables that differ in their low bits while both believing the table was
    "the fixed one". These offsets are part of the model's identity across the
    fleet, and identity must not depend on a numerical library.

    Rows are unit basis vectors spread evenly across the width, so they are
    exactly orthonormal in floating point on any platform. The refinement block
    is free to learn any basis it prefers; what matters here is that the slots
    are distinguishable, equal in norm, and byte-identical everywhere.
    """
    if m_max > d_model:
        raise ValueError(
            f"cannot build {m_max} orthonormal rows in {d_model} dimensions")
    table = torch.zeros(m_max, d_model, dtype=torch.float32)
    stride = d_model // m_max
    for i in range(m_max):
        table[i, i * stride] = 1.0
    return table


class Workspace(nn.Module):
    """``num_slots`` slots seeded from a state vector, refined ``num_steps``
    times, read out as a single vector of width ``d_model``.

    Parameter count is independent of both ``num_slots`` and ``num_steps``.
    """

    def __init__(self, d_model: int, *, num_slots: int = 1, num_steps: int = 1,
                 m_max: int = 8, hidden_mult: int = 2) -> None:
        super().__init__()
        if not 1 <= num_slots <= m_max:
            raise ValueError(f"num_slots must be in [1, {m_max}], got {num_slots}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        self.d_model = d_model
        self.num_slots = num_slots
        self.num_steps = num_steps
        self.m_max = m_max

        # A buffer, not a Parameter: it must not train, and it must travel with
        # the checkpoint so a reload cannot silently reseed.
        self.register_buffer("offsets", canonical_offsets(m_max, d_model),
                             persistent=True)

        # Content-based routing with shared projections. Each is d_model x
        # d_model and so carries the same parameters whether one slot attends
        # or eight do.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        hidden = d_model * hidden_mult
        self.refine = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_refine = nn.LayerNorm(d_model)

    def initial(self, state: torch.Tensor) -> torch.Tensor:
        """``(batch, d_model)`` -> ``(batch, num_slots, d_model)``."""
        if state.dim() != 2 or state.size(-1) != self.d_model:
            raise ValueError(
                f"expected state of shape (batch, {self.d_model}), "
                f"got {tuple(state.shape)}")
        return state.unsqueeze(1) + self.offsets[: self.num_slots].unsqueeze(0)

    def _route(self, z: torch.Tensor) -> torch.Tensor:
        """Single-head attention across slots. Shapes are M-independent."""
        q, k, v = self.q_proj(z), self.k_proj(z), self.v_proj(z)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        return self.o_proj(torch.matmul(torch.softmax(scores, dim=-1), v))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """``(batch, d_model)`` -> ``(batch, d_model)``."""
        z = self.initial(state)
        for _ in range(self.num_steps):
            z = self.norm_attn(z + self._route(z))
            z = self.norm_refine(z + self.refine(z))
        # Mean over slots, with the fixed offset mean removed. Without that
        # correction the readout carries a systematic M-dependent term: the
        # mean of the first M offsets differs between M=1 and M=8, so the two
        # cells would differ in their input statistics as well as in their slot
        # count. The correction is a constant, not a learned parameter.
        return z.mean(dim=1) - self.offsets[: self.num_slots].mean(dim=0)

    def extra_repr(self) -> str:  # pragma: no cover
        return (f"d_model={self.d_model}, num_slots={self.num_slots}, "
                f"num_steps={self.num_steps}, m_max={self.m_max}")
