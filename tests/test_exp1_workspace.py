"""The 2x2 is only interpretable if M and K cost no parameters.

An earlier version of this work compared arms that were different
architectures, so the result measured capacity rather than mechanism. These
tests exist to make that failure impossible to reintroduce silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exp1.workspace import Workspace, canonical_offsets  # noqa: E402

D_MODEL = 32
CELLS = [(1, 1), (1, 8), (8, 1), (8, 8)]


def _learned(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("num_slots,num_steps", CELLS)
def test_parameter_count_identical_across_the_2x2(num_slots, num_steps):
    """Every cell must carry exactly the parameters of the baseline cell."""
    baseline = _learned(Workspace(D_MODEL, num_slots=1, num_steps=1))
    cell = _learned(Workspace(D_MODEL, num_slots=num_slots, num_steps=num_steps))
    assert cell == baseline, (
        f"cell (M={num_slots}, K={num_steps}) has {cell} learned parameters "
        f"against the baseline's {baseline}; the 2x2 would measure capacity")


def test_offsets_are_not_trainable():
    """A trainable offset table would make M cost parameters after all."""
    ws = Workspace(D_MODEL, num_slots=8, num_steps=1)
    assert "offsets" in dict(ws.named_buffers())
    assert "offsets" not in dict(ws.named_parameters())
    assert not ws.offsets.requires_grad


def test_small_arm_is_a_strict_prefix_of_the_large_arm():
    """Slot 0 of M=1 must be bit-identical to slot 0 of M=8.

    Drawing a fresh table per M would leave the arms differing in slot content
    as well as slot count, which is a second confound on top of the one the
    parameter test covers.
    """
    one = Workspace(D_MODEL, num_slots=1, num_steps=1)
    eight = Workspace(D_MODEL, num_slots=8, num_steps=1)
    assert torch.equal(one.offsets[:1], eight.offsets[:1])

    state = torch.randn(4, D_MODEL, generator=torch.Generator().manual_seed(0))
    assert torch.equal(one.initial(state)[:, 0], eight.initial(state)[:, 0])


def test_offsets_are_exactly_orthonormal():
    """Exactly, not approximately. The table is part of the model's identity
    across machines, so it must not carry decomposition round-off."""
    table = canonical_offsets(8, D_MODEL)
    assert torch.equal(table @ table.T, torch.eye(8))
    assert torch.equal(table.norm(dim=1), torch.ones(8))
    # Every entry is exactly 0.0 or 1.0, so no platform can round it differently.
    assert set(table.unique().tolist()) == {0.0, 1.0}


def test_offsets_are_built_without_any_rng():
    """An earlier version derived the table from seeded noise via
    ``torch.linalg.qr``. That made a LAPACK build part of the fleet's model
    identity: pinning the sign against ``diag(R)`` removes the sign ambiguity
    but not floating-point differences between builds, so two nodes could hold
    tables differing in their low bits while both believing the table was fixed.

    The construction must not consult the RNG at all, so global seed state
    cannot influence it.
    """
    torch.manual_seed(0)
    first = canonical_offsets(8, D_MODEL)
    torch.manual_seed(12345)
    _ = torch.randn(1000)
    assert torch.equal(first, canonical_offsets(8, D_MODEL))


def test_offsets_do_not_depend_on_the_torch_thread_count():
    """Two machines with different core counts must build the same table.

    torch fills large CPU tensors through ``at::parallel_for``, so a seeded
    ``normal_`` can produce different values at different thread counts. A
    cross-node numerics probe in this project shipped with exactly that defect:
    two machines built different weights while asserting they had not. The
    construction no longer uses the RNG, but the property is worth pinning.
    """
    original = torch.get_num_threads()
    try:
        tables = []
        for threads in (1, 2, 4, 8):
            torch.set_num_threads(threads)
            tables.append(canonical_offsets(8, 256))
        for table in tables[1:]:
            assert torch.equal(tables[0], table)
    finally:
        torch.set_num_threads(original)


@pytest.mark.parametrize("num_slots", [1, 2, 4, 8])
def test_readout_has_no_M_dependent_offset_term(num_slots):
    """The readout must not carry a systematic term that moves with M.

    The mean of the first M offsets differs between M=1 and M=8, so a plain
    mean over slots would make the cells differ in their input statistics as
    well as in their slot count -- a confound underneath the parameter-count
    one. With the constant correction applied, an identity refinement leaves
    the readout equal to the state it was seeded from, for every M.
    """
    ws = Workspace(D_MODEL, num_slots=num_slots, num_steps=1)
    state = torch.randn(3, D_MODEL, generator=torch.Generator().manual_seed(7))
    z0 = ws.initial(state)
    readout_of_initial = z0.mean(dim=1) - ws.offsets[:num_slots].mean(dim=0)
    assert torch.allclose(readout_of_initial, state, atol=1e-6)


def test_routing_is_learned_and_parameter_invariant():
    """Cross-slot routing is learned, and its projections do not scale with M.

    An earlier revision used a parameter-free mean, justified by the claim that
    learned routing would make parameter count scale with M. That claim was
    wrong: attention projections are d_model x d_model however many slots
    attend through them. A global mean is also a real information bottleneck,
    and a screen built on it risks a negative result that says more about the
    coupling than about the hypothesis.
    """
    one = Workspace(D_MODEL, num_slots=1, num_steps=1)
    eight = Workspace(D_MODEL, num_slots=8, num_steps=1)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(one, name).weight.shape == (D_MODEL, D_MODEL)
        assert getattr(one, name).weight.shape == getattr(eight, name).weight.shape
    assert _learned(one) == _learned(eight)


@pytest.mark.parametrize("num_slots,num_steps", CELLS)
def test_output_shape_is_independent_of_the_cell(num_slots, num_steps):
    """Readout width must not move with M, or downstream layers would differ."""
    ws = Workspace(D_MODEL, num_slots=num_slots, num_steps=num_steps)
    out = ws(torch.randn(5, D_MODEL))
    assert out.shape == (5, D_MODEL)


def test_steps_are_weight_tied_not_stacked():
    """K must buy serial depth, not more weights.

    Asserted structurally rather than by parameter count alone: a stacked
    implementation could coincidentally match counts at some widths, and the
    hypothesis under test is specifically that iteration -- the same operator
    applied repeatedly -- is what helps.
    """
    ws = Workspace(D_MODEL, num_slots=1, num_steps=8)
    linears = [m for m in ws.refine if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 2, "refinement block should not grow with num_steps"


def test_more_steps_change_the_output():
    """Weight tying must not make extra steps a no-op."""
    torch.manual_seed(0)
    one = Workspace(D_MODEL, num_slots=4, num_steps=1)
    eight = Workspace(D_MODEL, num_slots=4, num_steps=8)
    eight.load_state_dict(one.state_dict())
    state = torch.randn(3, D_MODEL)
    assert not torch.allclose(one(state), eight(state))


def test_rejects_impossible_configurations():
    with pytest.raises(ValueError):
        Workspace(D_MODEL, num_slots=0)
    with pytest.raises(ValueError):
        Workspace(D_MODEL, num_slots=9, m_max=8)
    with pytest.raises(ValueError):
        Workspace(D_MODEL, num_steps=0)
    with pytest.raises(ValueError):
        canonical_offsets(64, 32)
