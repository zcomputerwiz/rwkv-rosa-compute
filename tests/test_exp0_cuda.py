"""CUDA-only validation for Experiment 0 training and recurrent kernels."""

import copy
import os
import random

import pytest
import torch
import torch.nn.functional as F

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, build_default_vocab, generate_packed_instances
from exp0.models.rwkv import RWKV7_OP, rwkv7_reference_step
from exp0.models.rwkv_cuda import rwkv7_cuda_recurrence, rwkv7_cuda_step
from exp0.train import create_model, train_model

pytestmark = [pytest.mark.exp0, pytest.mark.cuda]


@pytest.fixture(autouse=True)
def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _tiny_datasets(seed: int = 123):
    task_cfg = Task3SumConfig(
        length=6,
        dimension=2,
        num_filler=2,
        num_samples=16,
    )
    vocab = build_default_vocab(length=6, dimension=2)
    train_instances = generate_packed_instances(
        16,
        length=6,
        dimension=2,
        rng=random.Random(seed),
    )
    val_instances = generate_packed_instances(
        8,
        length=6,
        dimension=2,
        rng=random.Random(seed + 1),
    )
    train_ds = Task3SumDataset(
        train_instances,
        format_type="filler",
        num_filler=2,
        vocab=vocab,
        seed=seed,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        num_filler=2,
        vocab=vocab,
        seed=seed + 1,
    )
    return task_cfg, train_ds, val_ds


@pytest.mark.parametrize(
    ("precision", "fused_adamw"),
    [
        ("fp32", False),
        ("bf16", False),
        ("fp16", False),
        ("bf16", True),
    ],
)
def test_llama_cuda_training_smoke(precision, fused_adamw):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    task_cfg, train_ds, val_ds = _tiny_datasets()
    model_cfg = ModelConfig(
        architecture="llama",
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        device="cuda",
    )
    train_cfg = TrainConfig(
        batch_size=8,
        epochs=1,
        num_workers=0,
        val_num_workers=0,
        pin_memory=True,
        precision=precision,
        fused_adamw=fused_adamw,
        mixture="filler",
    )

    _, history = train_model(
        model_cfg,
        train_cfg,
        task_cfg,
        train_ds,
        val_ds,
    )

    assert len(history["epoch_train_losses"]) == 1
    assert torch.isfinite(torch.tensor(history["epoch_train_losses"][0]))
    assert history["epoch_seconds"][0] > 0
    assert history["samples_per_second"] > 0
    assert history["cuda_peak_memory_allocated_bytes"] > 0
    assert history["cuda_peak_memory_reserved_bytes"] > 0
    assert history["loss_reporting_syncs_per_epoch"] == 1
    assert history["validation_result_syncs_per_pass"] == 1
    assert history["non_blocking_transfers"] is True


def test_cuda_answer_only_projection_matches_full_prediction():
    torch.manual_seed(9)
    model = create_model(
        ModelConfig(
            architecture="llama",
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=31,
            device="cuda",
        ),
        d_input=10,
    ).cuda()
    model.eval()

    input_tuples = torch.randn(3, 4, 10, device="cuda")
    targets = torch.randint(0, 31, (3, 7), device="cuda")
    answer_positions = torch.tensor([0, 3, 5], device="cuda")

    with torch.no_grad():
        full_logits = model(input_tuples, targets)
        answer_logits = model.answer_logits(
            input_tuples,
            targets,
            answer_positions,
        )

    expected = full_logits[
        torch.arange(full_logits.shape[0], device="cuda"),
        answer_positions,
    ]
    assert (answer_logits - expected).abs().max().item() < 1e-5
    assert torch.equal(answer_logits.argmax(-1), expected.argmax(-1))


def test_llama_bf16_flash_sdpa_smoke():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        pytest.skip("This PyTorch build does not expose sdpa_kernel")

    # sdpa_kernel is exclusive: restricting to FLASH_ATTENTION also disables
    # EFFICIENT_ATTENTION and MATH. PyTorch's Windows wheels ship without the
    # flash backend compiled in, so that leaves zero backends and raises
    # "No available kernel". Probe before asserting anything about flash.
    probe = torch.randn(1, 4, 8, 64, device="cuda", dtype=torch.bfloat16)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(probe, probe, probe, is_causal=True)
    except RuntimeError as exc:
        pytest.skip(
            "Flash Attention backend unavailable in this PyTorch build "
            f"(Windows wheels ship without it): {exc}"
        )

    model = create_model(
        ModelConfig(
            architecture="llama",
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=64,
            device="cuda",
        ),
        d_input=12,
    ).cuda()
    input_tuples = torch.randn(2, 6, 12, device="cuda")
    targets = torch.randint(0, 64, (2, 12), device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            logits = model.loss_logits(input_tuples, targets)
            loss = logits.float().square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def _reference_from_raw_w(r, raw_w, k, v, a, b):
    transformed_w = -F.softplus(-raw_w) - 0.5
    return RWKV7_OP(r, transformed_w, k, v, a, b, head_dim=64)


def _require_cuda_toolkit():
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        if os.environ.get("EXP0_REQUIRE_RWKV_CUDA") == "1":
            pytest.fail("CUDA toolkit/nvcc is required for RWKV fused-kernel test")
        pytest.skip(
            "CUDA toolkit/nvcc is unavailable; set EXP0_REQUIRE_RWKV_CUDA=1 "
            "to make this a hard failure"
        )


def _step_inputs(timesteps):
    shape = (timesteps, 1, 768)
    r = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    raw_w = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    a = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    return r, raw_w, k, v, a, b


@pytest.mark.slow
@pytest.mark.parametrize("timesteps", [1, 2, 4, 16, 32, 128, 512])
def test_rwkv7_cuda_step_chain_matches_reference(timesteps):
    _require_cuda_toolkit()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    torch.manual_seed(741)
    inputs = _step_inputs(timesteps)
    reference_state = torch.zeros((1, 12, 64, 64), device="cuda")
    cuda_state = reference_state.clone()

    for timestep in range(timesteps):
        step_inputs = tuple(tensor[timestep] for tensor in inputs)
        reference_out, reference_state = rwkv7_reference_step(
            *step_inputs,
            reference_state,
        )
        cuda_out, returned_state = rwkv7_cuda_step(*step_inputs, cuda_state)
        assert returned_state.data_ptr() == cuda_state.data_ptr()
        torch.testing.assert_close(
            cuda_out.float(),
            reference_out,
            rtol=3e-2,
            atol=3e-2,
        )

    torch.testing.assert_close(
        cuda_state,
        reference_state,
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.slow
def test_rwkv7_cuda_step_cudagraph_matches_eager_reference():
    _require_cuda_toolkit()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    torch.manual_seed(742)
    inputs = tuple(tensor[0] for tensor in _step_inputs(1))
    state = torch.zeros((1, 12, 64, 64), device="cuda")

    rwkv7_cuda_step(*inputs, state)
    torch.cuda.synchronize()
    state.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_state = rwkv7_cuda_step(*inputs, state)

    state.zero_()
    graph.replay()
    torch.cuda.synchronize()
    reference_out, reference_state = rwkv7_reference_step(
        *inputs,
        torch.zeros_like(state),
    )

    assert graph_state.data_ptr() == state.data_ptr()
    torch.testing.assert_close(
        graph_out.float(),
        reference_out,
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(
        state,
        reference_state,
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.slow
@pytest.mark.parametrize("timesteps", [1, 2, 4, 16, 17, 32])
def test_rwkv7_cuda_full_model_sequence_matches_persistent_steps(timesteps):
    _require_cuda_toolkit()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    torch.manual_seed(1702)
    model = create_model(
        ModelConfig(
            architecture="rwkv",
            rwkv_kernel="cuda",
            hidden_size=768,
            num_hidden_layers=2,
            intermediate_size=3072,
            head_dim=64,
            vocab_size=256,
            device="cuda",
        ),
        d_input=64,
    ).cuda().bfloat16()
    for layer in model.backbone.layers:
        torch.nn.init.normal_(layer.time_mix.output.weight, std=0.01)
        torch.nn.init.normal_(layer.channel_mix.value.weight, std=0.01)

    targets = torch.randint(0, 256, (1, timesteps), device="cuda")
    tuples = torch.empty((1, 0, 64), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        sequence_logits = model(tuples, targets)
        state = model.init_rwkv_step_state(activation_dtype=torch.bfloat16)
        step_logits = []
        for timestep in range(timesteps):
            logits, state = model.rwkv_step(targets[:, timestep], state)
            step_logits.append(logits)

    torch.testing.assert_close(
        sequence_logits.float(),
        torch.stack(step_logits, dim=1).float(),
        rtol=5e-2,
        atol=5e-2,
    )


@pytest.mark.slow
def test_rwkv7_fused_cuda_forward_matches_reference_with_tail_padding():
    _require_cuda_toolkit()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    torch.manual_seed(77)
    shape = (2, 17, 64)  # deliberately not divisible by the kernel chunk length
    tensors = [
        (torch.randn(*shape, device="cuda") * 0.1).requires_grad_(True)
        for _ in range(6)
    ]
    r, raw_w, k, v, a, b = tensors

    reference = _reference_from_raw_w(r, raw_w, k, v, a, b)
    fused = rwkv7_cuda_recurrence(
        r,
        raw_w,
        k,
        v,
        a,
        b,
        head_dim=64,
    )

    assert fused.shape == reference.shape
    assert fused.is_contiguous()
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(
        fused.float(),
        reference.float(),
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.slow
def test_rwkv7_fused_cuda_backward_tracks_reference():
    _require_cuda_toolkit()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    torch.manual_seed(91)
    shape = (1, 17, 64)
    reference_inputs = [
        (torch.randn(*shape, device="cuda") * 0.05).requires_grad_(True)
        for _ in range(6)
    ]
    fused_inputs = [x.detach().clone().requires_grad_(True) for x in reference_inputs]
    probe = torch.randn(*shape, device="cuda") * 0.05

    ref_out = _reference_from_raw_w(*reference_inputs)
    (ref_out * probe).sum().backward()

    fused_out = rwkv7_cuda_recurrence(
        *fused_inputs,
        head_dim=64,
    )
    (fused_out.float() * probe).sum().backward()

    for ref_tensor, fused_tensor in zip(reference_inputs, fused_inputs):
        assert ref_tensor.grad is not None
        assert fused_tensor.grad is not None
        assert torch.isfinite(fused_tensor.grad).all()
        torch.testing.assert_close(
            fused_tensor.grad.float(),
            ref_tensor.grad.float(),
            rtol=8e-2,
            atol=8e-2,
        )


@pytest.mark.slow
def test_rwkv7_fused_cuda_full_model_forward_backward():
    """Exercise kernel integration through the whole Experiment 0 wrapper."""
    _require_cuda_toolkit()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    torch.manual_seed(1234)
    model = create_model(
        ModelConfig(
            architecture="rwkv",
            init_mode="random",
            rwkv_kernel="cuda",
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=128,
            head_dim=64,
            vocab_size=64,
            device="cuda",
        ),
        d_input=12,
    ).cuda()
    # 6 tuple positions + 11 target positions gives a 17-step backbone sequence,
    # deliberately exercising the fused wrapper's chunk-tail padding.
    input_tuples = torch.randn(2, 6, 12, device="cuda")
    targets = torch.randint(0, 64, (2, 11), device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.loss_logits(input_tuples, targets)
        loss = logits.float().square().mean()
    loss.backward()

    assert logits.shape == (2, 10, 64)
    assert torch.isfinite(loss)
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.slow
def test_rwkv7_cudagraph_full_model_matches_eager():
    """Keep the opt-in benchmark execution path behind the fused oracle."""
    _require_cuda_toolkit()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    torch.manual_seed(2026)
    eager_model = create_model(
        ModelConfig(
            architecture="rwkv",
            init_mode="random",
            rwkv_kernel="cuda",
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=128,
            head_dim=64,
            vocab_size=64,
            device="cuda",
        ),
        d_input=12,
    ).cuda()
    compiled_model = torch.compile(
        copy.deepcopy(eager_model),
        backend="cudagraphs",
        fullgraph=False,
        dynamic=False,
    )
    input_tuples = torch.randn(2, 6, 12, device="cuda")
    targets = torch.randint(0, 64, (2, 11), device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager_logits = eager_model(input_tuples, targets)
        eager_logits.float().square().mean().backward()
        compiled_logits = compiled_model(input_tuples, targets)
        compiled_logits.float().square().mean().backward()

    torch.testing.assert_close(
        compiled_logits.float(),
        eager_logits.float(),
        rtol=3e-2,
        atol=3e-2,
    )
    compared_gradients = 0
    for eager_parameter, compiled_parameter in zip(
        eager_model.parameters(), compiled_model.parameters()
    ):
        assert (compiled_parameter.grad is None) == (eager_parameter.grad is None)
        if eager_parameter.grad is None:
            continue
        torch.testing.assert_close(
            compiled_parameter.grad.float(),
            eager_parameter.grad.float(),
            rtol=8e-2,
            atol=8e-2,
        )
        compared_gradients += 1
    assert compared_gradients > 0
