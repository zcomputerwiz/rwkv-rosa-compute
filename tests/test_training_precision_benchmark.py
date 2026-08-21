"""CPU tests for the training-precision benchmark's non-CUDA logic."""

import pytest
import torch

from scripts.benchmark_training_precision import (
    SHAPES,
    VARIANTS,
    compiled_or_plain,
    set_tf32,
)

pytestmark = pytest.mark.exp0


def test_tf32_toggle_sets_every_related_backend_flag():
    set_tf32(True)
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.get_float32_matmul_precision() == "high"

    set_tf32(False)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"


def test_uncompiled_path_returns_the_bound_method():
    model = torch.nn.Linear(4, 4)
    model.loss_logits = lambda *a: None
    assert compiled_or_plain(model, compiled=False) is model.loss_logits


def test_compiling_a_module_would_bypass_loss_logits():
    """Regression for a silent measurement bug.

    torch.compile(model) wraps forward only; OptimizedModule forwards every
    other attribute to the original module. Compiling the module and then
    calling .loss_logits() therefore runs eager and reports a misleading 1.00x,
    which is why the benchmark compiles the bound method instead.
    """
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.layer(x)

        def loss_logits(self, x):
            return self.layer(x) * 2

    model = Model()
    optimized = torch.compile(model)
    assert optimized.loss_logits.__func__ is Model.loss_logits


def test_shapes_match_the_documented_experiment_runs():
    assert SHAPES["llama"]["batch"] == 384
    assert SHAPES["llama"]["layers"] == 4
    assert SHAPES["rwkv"]["batch"] == 128
    assert SHAPES["rwkv"]["layers"] == 12
    # The RWKV fused kernel requires bf16, so no fp32 variant may be offered.
    assert all(precision == "bf16" for _, precision, _, _ in VARIANTS["rwkv"])


def test_llama_variants_start_from_the_protocol_baseline():
    label, precision, tf32, compiled = VARIANTS["llama"][0]
    assert (label, precision, tf32, compiled) == ("fp32", "fp32", False, False)


# --- execution protocols are recorded, not silent -----------------------------

def test_strict_fp32_remains_the_default():
    """--precision fp32 must keep meaning strict FP32, not TF32."""
    from exp0.config import TrainConfig

    config = TrainConfig()
    assert config.tf32_matmul is False
    assert config.torch_compile is False


def test_execution_protocols_are_identity_neutral_at_default():
    from dataclasses import replace

    from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
    from exp0.evaluate import canonical_run_config

    model_cfg, task_cfg = ModelConfig(), Task3SumConfig()
    base = TrainConfig()
    baseline = canonical_run_config(model_cfg, base, task_cfg, 9999, 100, [42])
    assert "tf32_matmul" not in baseline["training_protocol"]
    assert "torch_compile" not in baseline["training_protocol"]

    for field in ("tf32_matmul", "torch_compile"):
        enabled = canonical_run_config(
            model_cfg, replace(base, **{field: True}), task_cfg, 9999, 100, [42])
        assert enabled["training_protocol"][field] is True
        assert enabled != baseline, f"{field} must change run identity"


def test_enabling_tf32_and_compile_are_distinct_identities():
    from dataclasses import replace

    from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
    from exp0.evaluate import compute_run_id

    model_cfg, task_cfg, base = ModelConfig(), Task3SumConfig(), TrainConfig()
    ids = {
        compute_run_id(model_cfg, cfg, task_cfg, 9999, 100, [42])
        for cfg in (base,
                    replace(base, tf32_matmul=True),
                    replace(base, torch_compile=True),
                    replace(base, tf32_matmul=True, torch_compile=True))
    }
    assert len(ids) == 4, "each protocol combination must be its own experiment"
