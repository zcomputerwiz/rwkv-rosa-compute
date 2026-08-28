"""Latent workspace: ``M`` addressable slots refined for ``K`` steps.

The experiment this exists for is a 2x2 over ``(M, K)``. That design only
measures what it claims to measure if moving along either axis changes *what
the model can do* and not *how many parameters it has*. An earlier version of
this line of work failed exactly there: the two arms carried different
parameter counts, so the comparison measured capacity rather than mechanism.

Two invariants therefore hold by construction, and are asserted by the tests:

``M`` costs no parameters
    Slot ``i`` is seeded as ``Z_i = S + c_i`` where the ``c_i`` are fixed,
    non-trainable, mutually orthonormal offsets. They are rows of a single
    seeded ``m_max x d_model`` table, and a workspace with ``M`` slots takes
    the *first* ``M`` rows. So slot 0 of ``M=1`` is bit-identical to slot 0 of
    ``M=8`` -- the small arm is a strict prefix of the large one, not a
    differently-seeded object.

``K`` costs no parameters
    The refinement block is weight-tied across steps. Running it eight times is
    the same weights applied eight times, so ``K`` buys serial depth and
    nothing else. This is the whole hypothesis: if extra steps help, they help
    because of iteration, not because of capacity.

Cross-slot communication is deliberately parameter-free (a mean over slots).
Anything learned there would scale with ``M`` and reintroduce the confound.
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_OFFSET_SEED = 20260828


def orthonormal_offsets(m_max: int, d_model: int,
                        seed: int = DEFAULT_OFFSET_SEED) -> torch.Tensor:
    """``m_max x d_model`` table with orthonormal rows, from a fixed seed.

    Generated once at full size and sliced, never regenerated per ``M``: a
    table drawn for ``M=1`` would not agree with the first row of a table drawn
    for ``M=8``, and the two arms would then differ in their slot-0 content as
    well as in their slot count.
    """
    if m_max > d_model:
        raise ValueError(
            f"cannot build {m_max} orthonormal rows in {d_model} dimensions")
    gen = torch.Generator().manual_seed(seed)
    raw = torch.empty(d_model, m_max, dtype=torch.float32).normal_(generator=gen)
    # QR gives orthonormal columns; transpose to rows. Sign is unspecified
    # across LAPACK builds, so pin it to the diagonal of R and keep the table
    # reproducible across machines.
    q, r = torch.linalg.qr(raw)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.T.contiguous()


class Workspace(nn.Module):
    """``num_slots`` slots seeded from a state vector, refined ``num_steps``
    times, read out as a single vector of width ``d_model``.

    Parameter count is independent of both ``num_slots`` and ``num_steps``.
    """

    def __init__(self, d_model: int, *, num_slots: int = 1, num_steps: int = 1,
                 m_max: int = 8, hidden_mult: int = 2,
                 offset_seed: int = DEFAULT_OFFSET_SEED) -> None:
        super().__init__()
        if not 1 <= num_slots <= m_max:
            raise ValueError(f"num_slots must be in [1, {m_max}], got {num_slots}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        self.d_model = d_model
        self.num_slots = num_slots
        self.num_steps = num_steps
        self.m_max = m_max

        # Registered as a buffer, not a Parameter: it must not train, and it
        # must travel with the checkpoint so a reload cannot silently reseed.
        self.register_buffer(
            "offsets", orthonormal_offsets(m_max, d_model, offset_seed),
            persistent=True)

        # One block, applied num_steps times. Input is the slot concatenated
        # with the cross-slot mean, so a slot can condition on the workspace as
        # a whole without any parameter that scales with M.
        hidden = d_model * hidden_mult
        self.refine = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def initial(self, state: torch.Tensor) -> torch.Tensor:
        """``(batch, d_model)`` -> ``(batch, num_slots, d_model)``."""
        if state.dim() != 2 or state.size(-1) != self.d_model:
            raise ValueError(
                f"expected state of shape (batch, {self.d_model}), "
                f"got {tuple(state.shape)}")
        return state.unsqueeze(1) + self.offsets[: self.num_slots].unsqueeze(0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """``(batch, d_model)`` -> ``(batch, d_model)``."""
        z = self.initial(state)
        for _ in range(self.num_steps):
            pooled = z.mean(dim=1, keepdim=True).expand_as(z)
            z = self.norm(z + self.refine(torch.cat([z, pooled], dim=-1)))
        # Mean over slots: parameter-free, and defined identically whatever M
        # is, so the readout does not change shape between cells.
        return z.mean(dim=1)

    def extra_repr(self) -> str:  # pragma: no cover
        return (f"d_model={self.d_model}, num_slots={self.num_slots}, "
                f"num_steps={self.num_steps}, m_max={self.m_max}")
