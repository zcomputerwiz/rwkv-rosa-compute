"""Equivalence tests for RWKV-7 time-mixing PyTorch forward pass vs numpy oracle and recurrent state resets."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from exp0.models.rwkv import RWKV7_OP, RWKV7_OP_step, RWKV7Backbone, RWKV7TimeMix


def numpy_time_mix_step(x, v_first, last_x, S, params, n_head=1, head_size=64):
    """Numpy oracle for single-token RWKV-7 time mixing step using dict params and per-head GroupNorm."""
    mr = params["mr"]
    mw = params["mw"]
    mk = params["mk"]
    mv = params["mv"]
    ma = params["ma"]
    mg = params["mg"]

    xr, xw, xk, xv, xa, xg = [x + m * (last_x - x) for m in [mr, mw, mk, mv, ma, mg]]

    r = params["Wr"] @ xr

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def softplus(z):
        return np.log1p(np.exp(z))

    w = np.exp(-np.exp(-softplus(-(params["w_bias"] + np.tanh(xw @ params["Ww1"]) @ params["Ww2"])) - 0.5))

    k = params["Wk"] @ xk
    v = params["Wv"] @ xv
    if v_first is None:
        v_first = v
    else:
        v = v + (v_first - v) * sigmoid(params["v0"] + (xv @ params["Wv1"]) @ params["Wv2"])

    a = sigmoid(params["a_bias"] + (xa @ params["Wa1"]) @ params["Wa2"])
    g = sigmoid(xg @ params["Wg1"]) @ params["Wg2"]
    kk = k * params["k_k"]
    k = k * (1 + (a - 1) * params["k_a"])

    r_reshaped, w_reshaped, k_reshaped, v_reshaped, kk_reshaped, a_reshaped = [
        i.reshape(n_head, head_size, 1) for i in [r, w, k, v, kk, a]
    ]
    r_k_reshaped = params["r_k"].reshape(n_head, head_size, 1)

    norm_kk = np.linalg.norm(kk_reshaped, axis=1, keepdims=True)
    kk_norm = kk_reshaped / np.maximum(norm_kk, 1e-12)

    sab = S @ (-kk_norm) @ (kk_norm * a_reshaped).swapaxes(-1, -2)
    S = S * w_reshaped.swapaxes(-1, -2) + sab + v_reshaped @ k_reshaped.swapaxes(-1, -2)
    y = S @ r_reshaped

    y_heads = y.reshape(n_head, head_size)
    mean = y_heads.mean(axis=1, keepdims=True)
    var = y_heads.var(axis=1, keepdims=True)
    y_norm = ((y_heads - mean) / np.sqrt(var + 64e-5)).reshape(-1) * params["ln_w"] + params["ln_b"]

    bonus = ((r_reshaped * k_reshaped * r_k_reshaped).sum(axis=1, keepdims=True) * v_reshaped).flatten()
    y_norm += bonus

    out = params["Wo"] @ (y_norm * g)
    return out, v_first, x, S


@pytest.mark.exp0
@pytest.mark.parametrize(
    ("C", "head_dim", "layer_id"),
    [
        (64, 64, 0),
        (128, 64, 0),
        (384, 64, 0),
        (384, 64, 1),
    ],
)
def test_pytorch_vs_numpy_oracle(C, head_dim, layer_id):
    """F3.1: Assert PyTorch RWKV7TimeMix matches numpy oracle across sequence steps and multi-head settings."""
    n_head = C // head_dim
    torch.manual_seed(42)

    tm = RWKV7TimeMix(hidden_size=C, layer_id=layer_id, num_layers=2, head_dim=head_dim)
    nn.init.orthogonal_(tm.output.weight)

    with torch.no_grad():
        tm.w1.normal_(0, 0.1)
        tm.a1.normal_(0, 0.1)
        tm.g1.normal_(0, 0.1)
        tm.v1.normal_(0, 0.1)

    params = {
        "mr": tm.x_r.detach().squeeze().numpy(),
        "mw": tm.x_w.detach().squeeze().numpy(),
        "mk": tm.x_k.detach().squeeze().numpy(),
        "mv": tm.x_v.detach().squeeze().numpy(),
        "ma": tm.x_a.detach().squeeze().numpy(),
        "mg": tm.x_g.detach().squeeze().numpy(),
        "w_bias": tm.w0.detach().squeeze().numpy(),
        "r_k": tm.r_k.detach().numpy(),
        "Ww1": tm.w1.detach().numpy(),
        "Ww2": tm.w2.detach().numpy(),
        "Wa1": tm.a1.detach().numpy(),
        "Wa2": tm.a2.detach().numpy(),
        "a_bias": tm.a0.detach().squeeze().numpy(),
        "Wg1": tm.g1.detach().numpy(),
        "Wg2": tm.g2.detach().numpy(),
        "v0": tm.v0.detach().squeeze().numpy(),
        "Wv1": tm.v1.detach().numpy(),
        "Wv2": tm.v2.detach().numpy(),
        "k_k": tm.k_k.detach().squeeze().numpy(),
        "k_a": tm.k_a.detach().squeeze().numpy(),
        "Wr": tm.receptance.weight.detach().numpy(),
        "Wk": tm.key.weight.detach().numpy(),
        "Wv": tm.value.weight.detach().numpy(),
        "Wo": tm.output.weight.detach().numpy(),
        "ln_w": tm.ln_x.weight.detach().numpy(),
        "ln_b": tm.ln_x.bias.detach().numpy(),
    }

    for T in [1, 2, 8]:
        torch.manual_seed(100 + T)
        x_pt = torch.randn(1, T, C)
        v_first_pt = torch.randn(1, T, C) if layer_id > 0 else None

        with torch.no_grad():
            out_pt, _ = tm(x_pt, v_first_pt)
            out_pt_np = out_pt.squeeze(0).numpy()

        x_np_seq = x_pt.squeeze(0).numpy()
        v_first_np_seq = v_first_pt.squeeze(0).numpy() if layer_id > 0 else None
        out_np_list = []

        last_x_np = np.zeros(C, dtype=np.float32)
        S_np = np.zeros((n_head, head_dim, head_dim), dtype=np.float32)

        for t in range(T):
            x_t = x_np_seq[t]
            vf_t = v_first_np_seq[t] if layer_id > 0 else None
            out_t, _, last_x_np, S_np = numpy_time_mix_step(
                x_t, vf_t, last_x_np, S_np, params, n_head=n_head, head_size=head_dim
            )
            out_np_list.append(out_t)

        out_np_seq = np.stack(out_np_list, axis=0)

        max_diff = np.max(np.abs(out_pt_np - out_np_seq))
        assert max_diff < 1e-6, f"Max diff at C={C}, head_dim={head_dim}, layer_id={layer_id}, T={T}: {max_diff}"


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


@pytest.mark.exp0
def test_rwkv7_as_constructed_liveness():
    """R4: Assert that the module as constructed (zero-initialized output weight) becomes fully live by step 2."""
    torch.manual_seed(42)
    tm = RWKV7TimeMix(hidden_size=128, layer_id=1, num_layers=2, head_dim=64)

    x = torch.randn(2, 4, 128, requires_grad=True)
    v_first = torch.randn(2, 4, 128, requires_grad=True)
    target = torch.randn(2, 4, 128)

    optimizer = torch.optim.SGD(tm.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    out_step2 = None
    for step in range(3):
        optimizer.zero_grad()
        out, _ = tm(x, v_first)
        loss = criterion(out, target)
        loss.backward()

        if step == 2:
            out_step2 = out
            for name, param in tm.named_parameters():
                assert param.grad is not None, f"Parameter {name} gradient is None at step 2"
                assert param.grad.abs().max().item() > 0.0, f"Parameter {name} gradient is zero at step 2"

        optimizer.step()

    assert out_step2 is not None and out_step2.abs().max().item() > 0.0, "Module output must be non-zero by step 2"
