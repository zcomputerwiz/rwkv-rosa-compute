"""Tests for the Experiment 0 training, runner, and reporting harness."""

import json
import random
import sys
from unittest.mock import patch

import pytest

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, build_default_vocab, pad_collate_fn
from exp0.evaluate import (
    canonical_run_config,
    compile_experiment_report,
    compute_run_id,
)
from exp0.task3sum import generate_instance
from exp0.train import create_model
from scripts import run_experiment


def test_evaluation_readout_is_supervised():
    """The evaluated ANS position must also be supervised by the training loss."""
    rng = random.Random(42)
    length, dimension = 6, 3
    instances = [
        generate_instance(length=length, dimension=dimension, rng=rng)
        for _ in range(2)
    ]

    vocab = build_default_vocab(length=length, dimension=dimension)
    ans_token_id = vocab.token2id["ANS"]
    ans_true_id = vocab.token2id["True"]
    ans_false_id = vocab.token2id["False"]

    dataset = Task3SumDataset(
        instances,
        format_type="filler",
        num_filler=10,
        vocab=vocab,
        seed=42,
    )
    sample = dataset[0]

    targets = sample["targets"]
    ans_positions = (targets == ans_token_id).nonzero(as_tuple=True)[0]
    assert len(ans_positions) == 1
    ans_pos = ans_positions[0].item()

    assert ans_pos + 1 < len(targets)
    answer_label_token = targets[ans_pos + 1].item()
    assert answer_label_token in (ans_true_id, ans_false_id)

    batch = pad_collate_fn([sample])
    shift_targets = batch["loss_mask"][:, 1:]
    supervised_target = shift_targets[0, ans_pos].item()

    assert supervised_target != -100
    assert supervised_target == answer_label_token


def test_runner_training_wiring_uses_real_train_signature(tmp_path, monkeypatch):
    """Exercise the real runner -> train_model API seam without doing training."""
    from exp0.train import train_model as real_train_model

    argv = [
        "run_experiment.py",
        "--architecture",
        "llama",
        "--hidden_size",
        "16",
        "--num_hidden_layers",
        "1",
        "--num_attention_heads",
        "1",
        "--intermediate_size",
        "32",
        "--length",
        "4",
        "--dimension",
        "2",
        "--num_samples",
        "2",
        "--val_samples",
        "2",
        "--epochs",
        "0",
        "--seeds",
        "42",
        "--format_type",
        "filler",
        "--out_dir",
        str(tmp_path),
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with patch(
        "scripts.run_experiment.train_model",
        wraps=real_train_model,
    ) as train_spy:
        run_experiment.main()

    train_spy.assert_called_once()
    kwargs = train_spy.call_args.kwargs
    assert kwargs["filler_val_dataset"] is not None
    assert kwargs["cot_val_dataset"] is not None
    assert len(kwargs["filler_val_dataset"]) == 2
    assert len(kwargs["cot_val_dataset"]) == 2

    reports = list(tmp_path.glob("*.json"))
    assert len(reports) == 1
    with open(reports[0], encoding="utf-8") as f:
        report = json.load(f)
    assert report["initialization"]["mode"] == "random"


def test_create_model_honors_rwkv_head_dim_and_kernel():
    model_cfg = ModelConfig(
        architecture="rwkv",
        init_mode="random",
        rwkv_kernel="reference",
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=256,
        head_dim=32,
        device="cpu",
    )

    model = create_model(model_cfg, d_input=32)

    time_mix = model.backbone.layers[0].time_mix
    assert time_mix.head_dim == 32
    assert time_mix.num_heads == 4
    assert time_mix.rwkv_kernel == "reference"

    cuda_model = create_model(
        ModelConfig(
            architecture="rwkv",
            init_mode="random",
            rwkv_kernel="cuda",
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=256,
            head_dim=64,
            device="cuda",
        ),
        d_input=32,
    )
    assert cuda_model.backbone.layers[0].time_mix.rwkv_kernel == "cuda"


def _per_seed_results():
    return [
        {
            "seed": 42,
            "task_seed": 42,
            "training_seed": 42,
            "best_filler_accuracy": 0.8,
            "best_cot_accuracy": 0.7,
            "best_val_accuracy": 0.8,
        },
        {
            "seed": 43,
            "task_seed": 43,
            "training_seed": 43,
            "best_filler_accuracy": 0.9,
            "best_cot_accuracy": 0.8,
            "best_val_accuracy": 0.9,
        },
        {
            "seed": 44,
            "task_seed": 44,
            "training_seed": 44,
            "best_filler_accuracy": 0.85,
            "best_cot_accuracy": 0.75,
            "best_val_accuracy": 0.85,
        },
    ]


def test_compile_experiment_report():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    task_cfg = Task3SumConfig()
    realized_counts = {"parallel_cot": 50, "filler": 50}

    report = compile_experiment_report(
        model_cfg,
        train_cfg,
        task_cfg,
        _per_seed_results(),
        majority_class_baseline=0.5,
        realized_mixture_counts=realized_counts,
        eval_seed=123,
        val_samples=500,
    )

    assert report["metrics"]["filler_accuracy"] == pytest.approx(0.85)
    assert report["metrics"]["cot_accuracy"] == pytest.approx(0.75)
    assert report["metrics"]["mean_accuracy"] == pytest.approx(0.85)
    assert report["metrics"]["min_accuracy"] == 0.8
    assert report["metrics"]["max_accuracy"] == 0.9
    assert report["majority_class_baseline"] == 0.5
    assert report["realized_mixture_counts"] == realized_counts
    assert report["eval_seed"] == 123
    assert report["val_samples"] == 500
    assert report["seeds_run"] == [42, 43, 44]
    assert report["run_config"]["evaluation"]["seeds_run"] == [42, 43, 44]


def test_compile_experiment_report_provenance():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(seed=111)
    task_cfg = Task3SumConfig(seed=222)
    per_seed_results = _per_seed_results()[:2]

    initialization = {
        "mode": "pretrained",
        "pretrained_scope": "backbone_only",
        "checkpoint_path": "/models/rwkv.pth",
        "checkpoint_sha256": "a" * 64,
        "strict_backbone_load": True,
    }
    for result in per_seed_results:
        result["initialization"] = initialization

    report = compile_experiment_report(
        model_cfg,
        train_cfg,
        task_cfg,
        per_seed_results,
        majority_class_baseline=0.5,
        realized_mixture_counts={},
        eval_seed=123,
        val_samples=500,
    )

    assert "seed" not in report["task_config"]
    assert "seed" not in report["training_protocol"]
    assert report["per_seed_details"][0]["seed"] == 42
    assert report["per_seed_details"][0]["task_seed"] == 42
    assert report["per_seed_details"][0]["training_seed"] == 42
    assert report["initialization"] == initialization
    assert len(report["run_id"]) == 16


def test_compute_run_id_covers_full_model_configuration():
    model_cfg = ModelConfig(architecture="llama")
    train_cfg = TrainConfig(batch_size=32)
    task_cfg = Task3SumConfig(length=10)

    run_id = compute_run_id(
        model_cfg,
        train_cfg,
        task_cfg,
        eval_seed=123,
        val_samples=1000,
        seeds_run=[1, 2, 3],
    )
    reordered = compute_run_id(
        model_cfg,
        train_cfg,
        task_cfg,
        eval_seed=123,
        val_samples=1000,
        seeds_run=[3, 2, 1],
    )
    assert run_id == reordered

    changed_ffn = ModelConfig(architecture="llama", intermediate_size=3072)
    assert run_id != compute_run_id(
        changed_ffn,
        train_cfg,
        task_cfg,
        eval_seed=123,
        val_samples=1000,
        seeds_run=[1, 2, 3],
    )

    changed_head_dim = ModelConfig(architecture="llama", head_dim=32)
    assert run_id != compute_run_id(
        changed_head_dim,
        train_cfg,
        task_cfg,
        eval_seed=123,
        val_samples=1000,
        seeds_run=[1, 2, 3],
    )

    pretrained = ModelConfig(
        architecture="rwkv",
        init_mode="pretrained",
        rwkv_checkpoint="/machine/a/model.pth",
        rwkv_checkpoint_sha256="b" * 64,
    )
    same_checkpoint_elsewhere = ModelConfig(
        architecture="rwkv",
        init_mode="pretrained",
        rwkv_checkpoint="/machine/b/model.pth",
        rwkv_checkpoint_sha256="b" * 64,
    )
    assert compute_run_id(
        pretrained,
        train_cfg,
        task_cfg,
        123,
        1000,
        [1],
    ) == compute_run_id(
        same_checkpoint_elsewhere,
        train_cfg,
        task_cfg,
        123,
        1000,
        [1],
    )

    reference_rwkv = ModelConfig(
        architecture="rwkv",
        init_mode="random",
        rwkv_kernel="reference",
    )
    cuda_rwkv = ModelConfig(
        architecture="rwkv",
        init_mode="random",
        rwkv_kernel="cuda",
    )
    assert compute_run_id(
        reference_rwkv,
        train_cfg,
        task_cfg,
        123,
        1000,
        [1],
    ) != compute_run_id(
        cuda_rwkv,
        train_cfg,
        task_cfg,
        123,
        1000,
        [1],
    )


def test_canonical_run_config_is_stored_without_checkpoint_path():
    model_cfg = ModelConfig(
        architecture="rwkv",
        init_mode="pretrained",
        rwkv_checkpoint="/machine/model.pth",
        rwkv_checkpoint_sha256="c" * 64,
    )
    config = canonical_run_config(
        model_cfg,
        TrainConfig(),
        Task3SumConfig(),
        9999,
        2000,
        [42],
    )

    assert "rwkv_checkpoint" not in config["model"]
    assert config["model"]["rwkv_checkpoint_sha256"] == "c" * 64


def test_existing_report_skip_requires_full_config_match(tmp_path, capsys):
    report_path = tmp_path / "report.json"
    current_config = {"model": {"hidden_size": 384}, "evaluation": {"seed": 1}}
    report_path.write_text(
        json.dumps({"run_id": "collision", "run_config": current_config}),
        encoding="utf-8",
    )

    assert run_experiment._check_existing_report(report_path, current_config)
    captured = capsys.readouterr()
    assert "full run_config matches" in captured.out

    with pytest.raises(ValueError, match="full run_config does not match"):
        run_experiment._check_existing_report(
            report_path,
            {"model": {"hidden_size": 768}, "evaluation": {"seed": 1}},
        )
