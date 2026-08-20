import argparse
import sys
from pathlib import Path

import pytest

from scripts import run_experiment
from scripts.sweep_n import (
    build_runner_command,
    canonical_sweep_config,
    compute_sweep_id,
    summarize_run_report,
    sweep_execution_protocol,
)
from scripts.sweep_n import get_parser as get_sweep_parser


def _sweep_args(**overrides) -> argparse.Namespace:
    values = {
        "architecture": "llama",
        "init": "random",
        "rwkv_checkpoint": None,
        "rwkv_kernel": "reference",
        "hidden_size": 384,
        "num_hidden_layers": 4,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "head_dim": 64,
        "length": 12,
        "dimension": 3,
        "num_samples": 1000,
        "val_samples": 200,
        "eval_seed": 9999,
        "seeds": [42, 43, 44],
        "format_type": None,
        "parallel_ratio": 0.5,
        "filler_ratio": 0.5,
        "serial_ratio": 0.0,
        "vocab_reduction": True,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 3,
        "num_workers": 2,
        "val_num_workers": 0,
        "prefetch_factor": 1,
        "precision": "fp32",
        "fused_adamw": False,
        "pin_memory": True,
        "immediate_protocol": True,
        "device": "cpu",
        "out_dir": "results/sweeps",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_runner_command():
    args = _sweep_args()
    n = 16
    run_out = Path("results/test")

    cmd = build_runner_command(args, n, run_out)

    assert cmd[0] == sys.executable
    assert cmd[1] == "scripts/run_experiment.py"
    assert cmd[cmd.index("--init") + 1] == "random"
    assert cmd[cmd.index("--rwkv_kernel") + 1] == "reference"
    assert cmd[cmd.index("--num_filler") + 1] == str(n)
    assert cmd[cmd.index("--num_samples") + 1] == "1000"
    assert cmd[cmd.index("--val_samples") + 1] == "200"
    assert cmd[cmd.index("--eval_seed") + 1] == "9999"
    assert cmd[cmd.index("--learning_rate") + 1] == "0.0001"
    assert cmd[cmd.index("--num_workers") + 1] == "2"
    assert cmd[cmd.index("--val_num_workers") + 1] == "0"
    assert cmd[cmd.index("--prefetch_factor") + 1] == "1"
    assert cmd[cmd.index("--precision") + 1] == "fp32"
    assert "--seeds" in cmd
    assert "42" in cmd and "43" in cmd and "44" in cmd
    assert "--vocab_reduction" in cmd
    assert "--pin_memory" in cmd
    assert "--no-fused_adamw" in cmd

    assert "--vocab_size" not in cmd
    assert "--weight_decay" not in cmd
    assert "--num_train_samples" not in cmd
    assert "--num_val_samples" not in cmd
    assert "--lr" not in cmd


def test_sweep_default_learning_rate_matches_positive_control():
    args = get_sweep_parser().parse_args([])
    assert args.learning_rate == pytest.approx(1e-4)


def test_build_runner_command_forwards_checkpoint_and_kernel():
    args = _sweep_args(
        architecture="rwkv",
        init="pretrained",
        rwkv_checkpoint="models/rwkv.pth",
        rwkv_kernel="cuda",
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        num_attention_heads=12,
        precision="bf16",
        fused_adamw=True,
    )
    cmd = build_runner_command(args, 4, Path("results/test"))

    assert cmd[cmd.index("--init") + 1] == "pretrained"
    assert cmd[cmd.index("--rwkv_checkpoint") + 1] == "models/rwkv.pth"
    assert cmd[cmd.index("--rwkv_kernel") + 1] == "cuda"
    assert cmd[cmd.index("--precision") + 1] == "bf16"
    assert "--fused_adamw" in cmd


def test_llama_defaults_to_random_initialization():
    args = run_experiment.get_parser().parse_args(
        ["--architecture", "llama", "--device", "cpu"]
    )
    task_cfg, model_cfg, train_cfg = run_experiment.build_configs(args)

    assert model_cfg.init_mode == "random"
    assert model_cfg.rwkv_checkpoint is None
    assert model_cfg.rwkv_checkpoint_sha256 is None
    assert model_cfg.rwkv_kernel == "reference"
    assert model_cfg.llama_rope_theta == 10000.0
    assert task_cfg.include_separator_token is True
    assert train_cfg.precision == "fp32"
    assert train_cfg.fused_adamw is False
    assert train_cfg.val_num_workers == 0


def test_llama_rejects_rwkv_cuda_kernel():
    args = run_experiment.get_parser().parse_args(
        [
            "--architecture",
            "llama",
            "--rwkv_kernel",
            "cuda",
            "--device",
            "cpu",
        ]
    )
    with pytest.raises(ValueError, match="only valid for RWKV"):
        run_experiment.build_configs(args)


def test_rwkv_requires_pretrained_checkpoint_or_explicit_random():
    parser = run_experiment.get_parser()
    args = parser.parse_args(["--architecture", "rwkv", "--device", "cpu"])

    with pytest.raises(ValueError, match="requires a stock pretrained checkpoint"):
        run_experiment.build_configs(args)

    random_args = parser.parse_args(
        [
            "--architecture",
            "rwkv",
            "--init",
            "random",
            "--device",
            "cpu",
        ]
    )
    _, model_cfg, _ = run_experiment.build_configs(random_args)
    assert model_cfg.init_mode == "random"


def test_rwkv_cuda_kernel_requires_head_dim_64():
    parser = run_experiment.get_parser()
    args = parser.parse_args(
        [
            "--architecture",
            "rwkv",
            "--init",
            "random",
            "--rwkv_kernel",
            "cuda",
            "--head_dim",
            "32",
            "--hidden_size",
            "384",
        ]
    )
    with pytest.raises(ValueError, match="requires head_dim=64"):
        run_experiment.build_configs(args)


def test_pretrained_rwkv_checkpoint_is_explicit_and_hashed(tmp_path):
    checkpoint = tmp_path / "rwkv.pth"
    checkpoint.write_bytes(b"synthetic checkpoint identity")

    args = run_experiment.get_parser().parse_args(
        [
            "--architecture",
            "rwkv",
            "--rwkv_checkpoint",
            str(checkpoint),
            "--hidden_size",
            "768",
            "--num_hidden_layers",
            "12",
            "--intermediate_size",
            "3072",
            "--head_dim",
            "64",
            "--device",
            "cpu",
        ]
    )
    _, model_cfg, _ = run_experiment.build_configs(args)

    assert model_cfg.init_mode == "pretrained"
    assert model_cfg.rwkv_checkpoint == str(checkpoint.resolve())
    assert model_cfg.rwkv_checkpoint_sha256 is not None
    assert len(model_cfg.rwkv_checkpoint_sha256) == 64


def test_random_rwkv_rejects_checkpoint(tmp_path):
    checkpoint = tmp_path / "rwkv.pth"
    checkpoint.write_bytes(b"not used")
    args = run_experiment.get_parser().parse_args(
        [
            "--architecture",
            "rwkv",
            "--init",
            "random",
            "--rwkv_checkpoint",
            str(checkpoint),
        ]
    )

    with pytest.raises(ValueError, match="must not be combined"):
        run_experiment.build_configs(args)


def test_sweep_id_changes_with_scientific_configuration():
    base = _sweep_args()
    changed_ffn = _sweep_args(intermediate_size=3072)
    changed_seeds = _sweep_args(seeds=[42, 99])
    changed_precision = _sweep_args(precision="bf16")
    changed_kernel = _sweep_args(
        architecture="rwkv",
        init="random",
        rwkv_kernel="cuda",
        precision="bf16",
        device="cuda",
    )
    rwkv_reference = _sweep_args(
        architecture="rwkv",
        init="random",
        rwkv_kernel="reference",
        precision="bf16",
        device="cuda",
    )

    assert compute_sweep_id(base) != compute_sweep_id(changed_ffn)
    assert compute_sweep_id(base) != compute_sweep_id(changed_seeds)
    assert compute_sweep_id(base) != compute_sweep_id(changed_precision)
    assert compute_sweep_id(changed_kernel) != compute_sweep_id(rwkv_reference)


def test_sweep_forwards_immediate_protocol_suppression():
    """The N=0 arm is training-budget aligned only if suppression reaches it."""
    cmd = build_runner_command(
        _sweep_args(immediate_protocol=False), 0, Path("out")
    )
    assert "--no-immediate_protocol" in cmd
    assert "--immediate_protocol" not in cmd

    default_cmd = build_runner_command(_sweep_args(), 0, Path("out"))
    assert "--immediate_protocol" in default_cmd


def test_immediate_protocol_default_preserves_sweep_identity():
    base = _sweep_args()
    assert "immediate_protocol" not in canonical_sweep_config(base)
    suppressed = _sweep_args(immediate_protocol=False)
    assert canonical_sweep_config(suppressed)["immediate_protocol"] is False
    assert compute_sweep_id(base) != compute_sweep_id(suppressed)


def test_sweep_execution_protocol_is_durable_but_not_part_of_identity():
    default = _sweep_args()
    canonical = canonical_sweep_config(default)
    execution = sweep_execution_protocol(default)

    assert "immediate_protocol" not in canonical
    assert execution["immediate_protocol_enabled"] is True
    assert execution["n0_training_budget_relation"] == "published_immediate_protocol"
    assert "actual model compute" in execution["compute_note"]

    suppressed = sweep_execution_protocol(_sweep_args(immediate_protocol=False))
    assert suppressed["immediate_protocol_enabled"] is False
    assert (
        suppressed["n0_training_budget_relation"]
        == "same_requested_epochs_weight_decay_and_grad_clip"
    )
    assert "does not equalize FLOPs or runtime" in suppressed["compute_note"]


def test_sweep_summary_preserves_child_execution_provenance():
    report = {
        "run_id": "run-0",
        "metrics": {
            "filler_accuracy": 0.75,
            "fixed_budget_run": False,
            "immediate_protocol_applied_any_seed": True,
            "immediate_protocol_per_seed": [
                {
                    "enabled": True,
                    "applied": True,
                    "trigger": "num_filler == 0",
                    "epochs_requested": 5,
                    "epochs_effective": 25,
                    "weight_decay_requested": 0.01,
                    "weight_decay_effective": 0.1,
                    "grad_clip_requested": 1.0,
                    "grad_clip_effective": 0.5,
                }
            ],
            "early_stopping_per_seed": [
                {
                    "epochs_requested": 5,
                    "epochs_effective": 25,
                    "epochs_trained": 10,
                }
            ],
        },
    }

    summary = summarize_run_report(0, Path("n0/report.json"), report)

    assert summary["n"] == 0
    assert summary["run_id"] == "run-0"
    assert summary["immediate_protocol_applied_any_seed"] is True
    assert summary["epochs_requested_per_seed"] == [5]
    assert summary["epochs_effective_per_seed"] == [25]
    assert summary["epochs_trained_per_seed"] == [10]
    assert summary["immediate_protocol_per_seed"][0]["weight_decay_effective"] == 0.1
