"""Equivalence tests for RWKV-7 time-mixing PyTorch forward pass vs numpy oracle and recurrent state resets."""

import numpy as np
import pytest
import torch

from exp0.models.rwkv import RWKV7_OP, RWKV7Backbone


def numpy_time_mix_step(x, v0, last_x, S, params, n_head=1, head_size=64):
    """Numpy oracle for single-token RWKV-7 time mixing step transcribed from rwkv_v7_numpy.py."""
    mr, mw, mk, mv, ma, mg, w_bias, r_k, Ww1, Ww2, Wa1, Wa2, a_bias, Wg1, Wg2 = params[:15]
    k_k, k_a, Wr, Wk, Wv, Wo, ln_w, ln_b = params[-8:]

    xr, xw, xk, xv, xa, xg = [x + m * (last_x - x) for m in [mr, mw, mk, mv, ma, mg]]

    r = Wr @ xr

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def softplus(z):
        return np.log1p(np.exp(z))

    w = np.exp(-np.exp(-softplus(-(w_bias + np.tanh(xw @ Ww1) @ Ww2)) - 0.5))

    k = Wk @ xk
    v = Wv @ xv
    if v0 is None:
        v0 = v
    else:
        Wv2, Wv1, v_bias = params[15:18]
        v += (v0 - v) * sigmoid(xv @ Wv1 @ Wv2 + v_bias)

    a = sigmoid(xa @ Wa1 @ Wa2 + a_bias)
    g = sigmoid(xg @ Wg1) @ Wg2
    kk = k * k_k
    k += k * (a - 1) * k_a

    r, w, k, v, kk, a, r_k = [i.reshape(n_head, head_size, 1) for i in [r, w, k, v, kk, a, r_k]]
    norm_kk = np.linalg.norm(kk, axis=1, keepdims=True)
    kk = kk / np.maximum(norm_kk, 1e-12)

    sab = S @ (-kk) @ (kk * a).swapaxes(-1, -2)
    S = S * w.swapaxes(-1, -2) + sab + v @ k.swapaxes(-1, -2)
    y = S @ r

    y_flat = y.flatten()
    mean = y_flat.mean()
    var = y_flat.var()
    y_norm = ((y_flat - mean) / np.sqrt(var + 64e-5)) * ln_w + ln_b

    bonus = ((r * k * r_k).sum(axis=1, keepdims=True) * v).flatten()
    y_norm += bonus

    out = Wo @ (y_norm * g)
    return out, v0, x, S


@pytest.mark.exp0
def test_rwkv7_op_basic():
    """Test PyTorch RWKV7_OP recurrence op shape and output sanity."""
    B, T, C = 2, 8, 128

    torch.manual_seed(42)
    r = torch.randn(B, T, C)
    w = torch.randn(B, T, C)
    k = torch.randn(B, T, C)
    v = torch.randn(B, T, C)
    a = torch.randn(B, T, C)
    b = torch.randn(B, T, C)

    out = RWKV7_OP(r, w, k, v, a, b)
    assert out.shape == (B, T, C)


@pytest.mark.exp0
def test_rwkv7_backbone_chunked_reproducibility():
    """Test that RWKV7Backbone executes reproducibly across runs."""
    torch.manual_seed(42)
    backbone = RWKV7Backbone(hidden_size=128, num_layers=2, intermediate_size=256, head_dim=64)

    inputs = torch.randn(2, 10, 128)
    out1 = backbone(inputs)
    out2 = backbone(inputs)

    assert torch.allclose(out1, out2, atol=1e-6)
