import argparse
import sys
from pathlib import Path

from scripts.sweep_n import build_runner_command


def test_build_runner_command():
    args = argparse.Namespace(
        architecture="llama",
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=6,
        intermediate_size=1536,
        head_dim=64,
        length=12,
        dimension=3,
        num_samples=1000,
        val_samples=200,
        batch_size=64,
        learning_rate=1e-4,
        epochs=3,
        num_workers=2,
        device="cpu",
    )
    n = 16
    run_out = Path("results/test")

    cmd = build_runner_command(args, n, run_out)

    assert cmd[0] == sys.executable
    assert cmd[1] == "scripts/run_experiment.py"

    # Assert updated names are present
    assert "--num_filler" in cmd
    assert str(n) in cmd

    assert "--num_samples" in cmd
    assert "1000" in cmd

    assert "--val_samples" in cmd
    assert "200" in cmd

    assert "--learning_rate" in cmd
    assert "0.0001" in cmd

    assert "--num_workers" in cmd
    assert "2" in cmd

    # Assert stale names are absent
    assert "--vocab_size" not in cmd
    assert "--weight_decay" not in cmd
    assert "--num_train_samples" not in cmd
    assert "--num_val_samples" not in cmd
    assert "--lr" not in cmd
