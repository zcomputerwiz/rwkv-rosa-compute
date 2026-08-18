"""Equivalence tests for RWKV-7 time-mixing PyTorch forward pass vs numpy oracle and recurrent state resets."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from exp0.models.rwkv import RWKV7_OP, RWKV7_OP_step, RWKV7Backbone, RWKV7TimeMix


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
    elif len(params) > 23:
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
def test_pytorch_vs_numpy_oracle():
    """F3.1: Assert PyTorch RWKV7TimeMix matches numpy oracle across sequence steps."""
    torch.manual_seed(42)
    C = 64
    head_dim = 64
    n_head = 1

    tm = RWKV7TimeMix(hidden_size=C, layer_id=0, num_layers=1, head_dim=head_dim)
    nn.init.orthogonal_(tm.output.weight)

    # Extract parameters for numpy oracle
    mr = tm.x_r.detach().squeeze().numpy()
    mw = tm.x_w.detach().squeeze().numpy()
    mk = tm.x_k.detach().squeeze().numpy()
    mv = tm.x_v.detach().squeeze().numpy()
    ma = tm.x_a.detach().squeeze().numpy()
    mg = tm.x_g.detach().squeeze().numpy()

    w_bias = tm.w0.detach().squeeze().numpy()
    r_k = tm.r_k.detach().squeeze().numpy()
    Ww1 = tm.w1.detach().numpy()
    Ww2 = tm.w2.detach().numpy()
    Wa1 = tm.a1.detach().numpy()
    Wa2 = tm.a2.detach().numpy()
    a_bias = tm.a0.detach().squeeze().numpy()
    Wg1 = tm.g1.detach().numpy()
    Wg2 = tm.g2.detach().numpy()

    k_k = tm.k_k.detach().squeeze().numpy()
    k_a = tm.k_a.detach().squeeze().numpy()
    Wr = tm.receptance.weight.detach().numpy()
    Wk = tm.key.weight.detach().numpy()
    Wv = tm.value.weight.detach().numpy()
    Wo = tm.output.weight.detach().numpy()
    ln_w = tm.ln_x.weight.detach().numpy()
    ln_b = tm.ln_x.bias.detach().numpy()

    params = [
        mr, mw, mk, mv, ma, mg, w_bias, r_k, Ww1, Ww2, Wa1, Wa2, a_bias, Wg1, Wg2,
        k_k, k_a, Wr, Wk, Wv, Wo, ln_w, ln_b
    ]

    for T in [1, 2, 8]:
        torch.manual_seed(100 + T)
        x_pt = torch.randn(1, T, C)

        with torch.no_grad():
            out_pt, _ = tm(x_pt, None)
            out_pt_np = out_pt.squeeze(0).numpy()

        x_np_seq = x_pt.squeeze(0).numpy()
        out_np_list = []

        v0_np = None
        last_x_np = np.zeros(C, dtype=np.float32)
        S_np = np.zeros((n_head, head_dim, head_dim), dtype=np.float32)

        for t in range(T):
            x_t = x_np_seq[t]
            out_t, v0_np, last_x_np, S_np = numpy_time_mix_step(
                x_t, v0_np, last_x_np, S_np, params, n_head=n_head, head_size=head_dim
            )
            out_np_list.append(out_t)

        out_np_seq = np.stack(out_np_list, axis=0)

        max_diff = np.max(np.abs(out_pt_np - out_np_seq))
        assert max_diff < 1e-4, f"Max diff at T={T}: {max_diff}"


@pytest.mark.exp0
def test_rwkv7_time_mix_initialization_and_gradients():
    """F2: Assert RWKV7TimeMix output is non-zero, zero-initialized factors break fixed points, and every param receives gradient."""
    for layer_id in [0, 1]:
        torch.manual_seed(42)
        tm = RWKV7TimeMix(hidden_size=128, layer_id=layer_id, num_layers=2, head_dim=64)

        nn.init.orthogonal_(tm.output.weight)

        x = torch.randn(2, 4, 128, requires_grad=True)
        v_first = torch.randn(2, 4, 128, requires_grad=True) if layer_id > 0 else None

        out, _ = tm(x, v_first)
        assert out.abs().max().item() > 0.0, "Time-mix output must be non-zero"

        loss = out.sum()
        loss.backward()

        # Step 0: First factors w1, a1, g1 must receive non-zero gradients
        for name in ["w1", "a1", "g1", "k_k", "k_a", "r_k"]:
            param = dict(tm.named_parameters())[name]
            assert param.grad is not None, f"Parameter {name} gradient is None"
            assert param.grad.abs().max().item() > 0.0, f"Parameter {name} gradient is zero on step 0"

        # Take one optimizer step to update w1, a1, g1
        optimizer = torch.optim.SGD(tm.parameters(), lr=0.1)
        optimizer.step()

        # Step 1: Forward/backward after 1 step
        optimizer.zero_grad()
        out2, _ = tm(x, v_first)
        loss2 = out2.sum()
        loss2.backward()

        for name, param in tm.named_parameters():
            if layer_id == 0 and name in ["v0", "v1", "v2"]:
                continue
            assert param.grad is not None, f"Parameter {name} gradient is None after step 1 at layer_id={layer_id}"
            assert param.grad.abs().max().item() > 0.0, f"Parameter {name} gradient is zero after step 1 at layer_id={layer_id}"


@pytest.mark.exp0
def test_rwkv7_op_chunked_vs_step_recurrent():
    """F3.2: Assert chunked forward pass matches step-wise recurrent forward pass."""
    B, T, C = 2, 6, 128
    N = 64
    H = C // N

    torch.manual_seed(42)
    r = torch.randn(B, T, C)
    w = torch.randn(B, T, C)
    k = torch.randn(B, T, C)
    v = torch.randn(B, T, C)
    a = torch.randn(B, T, C)
    b = torch.randn(B, T, C)

    out_chunked = RWKV7_OP(r, w, k, v, a, b, head_dim=N)

    out_step_list = []
    state = torch.zeros((B, H, N, N), dtype=torch.float32)

    for t in range(T):
        out_t, state = RWKV7_OP_step(
            r[:, t, :],
            w[:, t, :],
            k[:, t, :],
            v[:, t, :],
            a[:, t, :],
            b[:, t, :],
            state,
            head_dim=N,
        )
        out_step_list.append(out_t)

    out_step = torch.stack(out_step_list, dim=1)
    assert torch.allclose(out_chunked, out_step, atol=1e-5), f"Max diff: {(out_chunked - out_step).abs().max()}"


@pytest.mark.exp0
def test_rwkv7_state_reset_clean_isolation():
    """F3.3: Assert that running sequence A then sequence B with state reset produces identical result as B alone."""
    torch.manual_seed(42)
    backbone = RWKV7Backbone(hidden_size=128, num_layers=2, intermediate_size=256, head_dim=64)

    seq_a = torch.randn(2, 8, 128)
    seq_b = torch.randn(2, 8, 128)

    out_b_alone = backbone(seq_b)

    _ = backbone(seq_a)
    out_b_after_a = backbone(seq_b)

    assert torch.allclose(out_b_alone, out_b_after_a, atol=1e-6), "State must reset cleanly between separate forward passes"
