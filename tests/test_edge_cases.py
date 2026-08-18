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
    assert ln[0] == 0
    assert ln[1] == 0
    assert idx[0] == 0
    assert idx[1] == 0


def test_latest_match_wins_ties(latest_match_fixture):
    f = latest_match_fixture
    idx, ln = rosa_slow_ref(f["q_sym"], f["k_sym"], f["v_sym"])
    assert idx == f["expected_idx"]
    assert ln == f["expected_ln"]


def test_longest_suffix_precedence(longest_match_fixture):
    f = longest_match_fixture
    idx, ln = rosa_slow_ref(f["q_sym"], f["k_sym"], f["v_sym"])
    assert idx == f["expected_idx"]
    assert ln == f["expected_ln"]


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
    # Output should be purely in {-1, 0, +1}
    unique_vals = set(out.unique().tolist())
    assert unique_vals.issubset({-1.0, 0.0, 1.0})
