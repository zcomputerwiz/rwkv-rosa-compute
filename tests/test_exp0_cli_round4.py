import argparse
import sys
from pathlib import Path

import pytest

from scripts import run_experiment
from scripts.sweep_n import build_runner_command, compute_sweep_id


def _sweep_args(**overrides) -> argparse.Namespace:
    values = {
        "architecture": "llama",
        "init": "random",
        "rwkv_checkpoint": None,
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
    assert cmd[cmd.index("--num_filler") + 1] == str(n)
    assert cmd[cmd.index("--num_samples") + 1] == "1000"
    assert cmd[cmd.index("--val_samples") + 1] == "200"
    assert cmd[cmd.index("--eval_seed") + 1] == "9999"
    assert cmd[cmd.index("--learning_rate") + 1] == "0.0001"
    assert cmd[cmd.index("--num_workers") + 1] == "2"
    assert "--seeds" in cmd
    assert "42" in cmd and "43" in cmd and "44" in cmd
    assert "--vocab_reduction" in cmd

    assert "--vocab_size" not in cmd
    assert "--weight_decay" not in cmd
    assert "--num_train_samples" not in cmd
    assert "--num_val_samples" not in cmd
    assert "--lr" not in cmd


def test_build_runner_command_forwards_checkpoint():
    args = _sweep_args(
        architecture="rwkv",
        init="pretrained",
        rwkv_checkpoint="models/rwkv.pth",
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        num_attention_heads=12,
    )
    cmd = build_runner_command(args, 4, Path("results/test"))

    assert cmd[cmd.index("--init") + 1] == "pretrained"
    assert cmd[cmd.index("--rwkv_checkpoint") + 1] == "models/rwkv.pth"


def test_llama_defaults_to_random_initialization():
    args = run_experiment.get_parser().parse_args(
        ["--architecture", "llama", "--device", "cpu"]
    )
    _, model_cfg, _ = run_experiment.build_configs(args)

    assert model_cfg.init_mode == "random"
    assert model_cfg.rwkv_checkpoint is None
    assert model_cfg.rwkv_checkpoint_sha256 is None


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

    assert compute_sweep_id(base) != compute_sweep_id(changed_ffn)
    assert compute_sweep_id(base) != compute_sweep_id(changed_seeds)
