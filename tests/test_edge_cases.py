import pytest
import torch

from rosa_compute import blinkdl_rosa_4bit_reference, rosa_slow_ref


def test_no_match():
    # Query symbol never appears in key
    q = [15, 15]
    k = [0, 1]
    v = [10, 20]
    idx, ln = rosa_slow_ref(q, k, v)
    assert ln[1] == 0

def test_latest_match_wins_ties():
    # Same suffix length at multiple positions
    # q=[5, 5, 0], k=[5, 5, 0], v=[10, 20, 30]
    # At i=1 (q[1]=5): w=1, t=[5].
    # j ranges from 0 down to 0: k[0:1]==[5] -> match! s = 0+1 = 1, idx[1]=v[1]=20, ln[1]=1.
    q = [5, 5, 0]
    k = [5, 5, 0]
    v = [10, 20, 30]
    idx, ln = rosa_slow_ref(q, k, v)
    assert idx[1] == 20
    assert ln[1] == 1

def test_longest_suffix_precedence():
    # Shorter match later vs longer match earlier
    q_seq = [1, 2, 1, 2]
    k_seq = [1, 2, 1, 2]
    v_seq = [10, 20, 30, 40]
    idx, ln = rosa_slow_ref(q_seq, k_seq, v_seq)
    assert ln[3] == 2
    assert idx[3] == 30

@pytest.mark.parametrize("T", [1, 2, 4, 8, 32, 33, 64, 512])
def test_context_length_boundaries(T):
    q = torch.randn(1, T, 16)
    k = torch.randn(1, T, 16)
    v = torch.randn(1, T, 16)
    out = blinkdl_rosa_4bit_reference(q, k, v)
    assert out.shape == (1, T, 16)
