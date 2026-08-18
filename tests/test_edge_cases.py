import pytest
import torch

from rosa_compute import blinkdl_rosa_4bit_reference, rosa_slow_ref


def test_no_match():
    """No Match Fixture:
    Query symbol never appears in key sequence.
    Q symbols: [15, 15]
    K symbols: [0, 1]
    V symbols: [10, 20]
    Result: match length 0, output 0.
    """
    q = [15, 15]
    k = [0, 1]
    v = [10, 20]
    idx, ln = rosa_slow_ref(q, k, v)
    assert ln == [0, 0]
    assert idx == [0, 0]


def test_latest_match_wins_ties(latest_match_fixture):
    """Verifies that when two distinct prior starting positions produce the same
    maximum suffix length w, the latest valid occurrence (j2 > j1) is selected.
    """
    f = latest_match_fixture
    idx, ln = rosa_slow_ref(f["q_sym"], f["k_sym"], f["v_sym"])
    assert idx == f["expected_idx"]
    assert ln == f["expected_ln"]


def test_longest_suffix_precedence(longest_match_fixture):
    """Verifies that a longer suffix match at an earlier position (j=0, w=2)
    takes precedence over a shorter suffix match at a later position (j=1, w=1).
    """
    f = longest_match_fixture
    idx, ln = rosa_slow_ref(f["q_sym"], f["k_sym"], f["v_sym"])
    assert idx == f["expected_idx"]
    assert ln == f["expected_ln"]


def test_route_index_offset(exact_match_fixture):
    """Verifies that the returned value is exactly V[j + w], where j is the
    matching route start index and w is the suffix match length.
    """
    f = exact_match_fixture
    idx, ln = rosa_slow_ref(f["q_sym"], f["k_sym"], f["v_sym"])
    assert idx == f["expected_idx"]
    assert ln == f["expected_ln"]
    # At i=2, j=0 and w=2 -> s = 0 + 2 = 2 -> V[2] = 30
    assert idx[2] == f["v_sym"][0 + 2]


def test_single_bit_mismatch():
    """Single-bit Difference Fixture:
    Two candidate symbols differ by exactly 1 bit.
    Symbol 1 = 0001 (1)
    Symbol 2 = 0000 (0)
    Verify single-bit difference prevents suffix matching.
    """
    q = [1]
    k = [0]
    v = [10]
    idx, ln = rosa_slow_ref(q, k, v)
    assert ln[0] == 0
    assert idx[0] == 0


@pytest.mark.parametrize("sym", [0, 1, 2, 4, 8, 15])
def test_all_four_bit_symbols(sym):
    """Test explicit 4-bit symbol encoding values: 0, 1, 2, 4, 8, 15."""
    q = [sym, sym]
    k = [sym, sym]
    v = [10, 20]
    idx, ln = rosa_slow_ref(q, k, v)
    assert ln[1] == 1
    assert idx[1] == 20


@pytest.mark.parametrize("T", [1, 2, 4, 8, 32, 33, 64, 128, 512])
def test_context_length_boundaries(T):
    q = torch.randn(1, T, 16)
    k = torch.randn(1, T, 16)
    v = torch.randn(1, T, 16)
    out = blinkdl_rosa_4bit_reference(q, k, v)
    assert out.shape == (1, T, 16)
    unique_vals = set(out.unique().tolist())
    assert unique_vals.issubset({-1.0, 0.0, 1.0})
