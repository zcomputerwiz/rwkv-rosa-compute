import random
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset
from exp0.evaluate import compile_experiment_report
from exp0.task3sum import generate_instance
from exp0.train import evaluate_accuracy, train_model

pytestmark = pytest.mark.exp0


def get_tiny_configs():
    task_cfg = Task3SumConfig(num_samples=16)
    model_cfg = ModelConfig(
        architecture="llama",
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        device="cpu",
    )
    train_cfg = TrainConfig(batch_size=8, epochs=1, num_workers=0)
    return task_cfg, model_cfg, train_cfg


def _generate_instances(task_cfg: Task3SumConfig, seed: int):
    rng = random.Random(seed)
    return [
        generate_instance(
            length=task_cfg.length,
            dimension=task_cfg.dimension,
            mod=task_cfg.mod,
            rng=rng,
        )
        for _ in range(task_cfg.num_samples)
    ]


def test_train_model_dual_validation():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()

    train_instances = _generate_instances(task_cfg, seed=100)
    val_instances = _generate_instances(task_cfg, seed=101)
    train_ds = Task3SumDataset(
        train_instances,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    cot_ds = Task3SumDataset(
        val_instances,
        format_type="parallel_cot",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )

    _, history = train_model(
        model_cfg,
        train_cfg,
        task_cfg,
        train_ds,
        val_ds,
        cot_val_dataset=cot_ds,
    )

    assert len(history["epoch_train_losses"]) == train_cfg.epochs
    assert len(history["epoch_online_train_answer_accuracies"]) == train_cfg.epochs
    assert len(history["epoch_filler_accuracies"]) == train_cfg.epochs
    assert len(history["epoch_cot_diagnostics"]) == train_cfg.epochs

    assert "best_online_train_answer_accuracy" in history
    assert "best_filler_accuracy" in history
    assert "best_cot_diagnostics" in history
    assert "cot_answer_given_cot_accuracy" in history["best_cot_diagnostics"]
    assert "cot_result_semantic_accuracy" in history["best_cot_diagnostics"]
    assert "cot_result_nll" in history["best_cot_diagnostics"]
    assert history["adam_betas"] == [0.9, 0.95]
    assert history["lr_schedule"] == "linear_warmup_decay"
    assert "best_cot_accuracy" not in history
    # Validation is filler-format only; the duplicate "val" aliases were removed
    # so that agreement between two report keys cannot be read as corroboration.
    assert "best_val_accuracy" not in history
    assert "epoch_val_accuracies" not in history
    assert "best_filler_answer_prediction_counts" in history
    counts = history["best_filler_answer_prediction_counts"]
    assert counts["total"] == (
        counts["predicted_true"]
        + counts["predicted_false"]
        + counts["predicted_other"]
    )
    assert isinstance(counts["degenerate_predictor"], bool)


def test_train_model_single_validation():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()

    train_instances = _generate_instances(task_cfg, seed=200)
    val_instances = _generate_instances(task_cfg, seed=201)
    train_ds = Task3SumDataset(
        train_instances,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )

    _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)

    assert "epoch_train_losses" in history
    assert "epoch_online_train_answer_accuracies" in history
    assert "epoch_filler_accuracies" in history
    assert "epoch_cot_diagnostics" not in history
    assert "best_cot_diagnostics" not in history


def test_missing_ids_fails():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    train_instances = _generate_instances(task_cfg, seed=300)
    val_instances = _generate_instances(task_cfg, seed=301)
    train_ds = Task3SumDataset(
        train_instances,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )

    if "ANS" in train_ds.vocab.token2id:
        del train_ds.vocab.token2id["ANS"]

    with pytest.raises(
        ValueError,
        match="Vocabulary must contain 'ANS', 'True', and 'False'",
    ):
        train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)


class MockModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.call_count = 0

    def forward(self, input_tuples, targets):
        batch_size = targets.size(0)
        out = self.logits[self.call_count : self.call_count + batch_size]
        self.call_count += batch_size
        return out


def test_evaluate_accuracy_exact_scoring():
    ans_token_id = 10
    ans_true_id = 1
    ans_false_id = 2
    unrelated_id = 3
    device = torch.device("cpu")

    targets = torch.tensor(
        [
            [0, 0, ans_token_id, ans_true_id, 0],
            [0, 0, ans_token_id, ans_false_id, 0],
            [0, 0, ans_token_id, ans_false_id, 0],
            [0, 0, ans_token_id, ans_true_id, 0],
        ]
    )
    has_3sum = torch.tensor([True, False, False, True])

    logits = torch.zeros(4, 5, 20)
    logits[0, 2, ans_true_id] = 10.0
    logits[1, 2, ans_false_id] = 10.0
    logits[2, 2, unrelated_id] = 10.0
    logits[3, 2, ans_false_id] = 10.0

    mock_model = MockModel(logits)
    mock_loader = [
        {
            "input_tuples": torch.zeros(4, 1, 1),
            "targets": targets,
            "has_3sum": has_3sum,
        }
    ]

    accuracy = evaluate_accuracy(
        mock_model,
        mock_loader,
        device,
        ans_token_id,
        ans_true_id,
        ans_false_id,
    )

    assert accuracy == 0.5


def test_evaluate_accuracy_malformed_ans():
    ans_token_id = 10
    ans_true_id = 1
    ans_false_id = 2
    device = torch.device("cpu")

    mock_model = MockModel(torch.zeros(1, 5, 20))

    targets_zero = torch.tensor([[0, 0, 0, 0, 0]])
    mock_loader_zero = [
        {
            "input_tuples": torch.zeros(1, 1, 1),
            "targets": targets_zero,
            "has_3sum": torch.tensor([True]),
        }
    ]

    with pytest.raises(ValueError, match="Sequence must have exactly one ANS token"):
        evaluate_accuracy(
            mock_model,
            mock_loader_zero,
            device,
            ans_token_id,
            ans_true_id,
            ans_false_id,
        )

    mock_model.call_count = 0
    targets_multiple = torch.tensor([[0, ans_token_id, ans_token_id, 0, 0]])
    mock_loader_multiple = [
        {
            "input_tuples": torch.zeros(1, 1, 1),
            "targets": targets_multiple,
            "has_3sum": torch.tensor([True]),
        }
    ]

    with pytest.raises(ValueError, match="Sequence must have exactly one ANS token"):
        evaluate_accuracy(
            mock_model,
            mock_loader_multiple,
            device,
            ans_token_id,
            ans_true_id,
            ans_false_id,
        )


def test_early_stop_target_resolution():
    from exp0.config import TrainConfig
    from exp0.train import early_stop_reached, resolve_early_stop_target

    acc_cfg = TrainConfig(early_stop_metric="filler_accuracy")
    assert resolve_early_stop_target(acc_cfg, None) == 1.0
    assert early_stop_reached(acc_cfg, 1.0, 1.0)
    assert not early_stop_reached(acc_cfg, 0.999, 1.0)

    tol_cfg = TrainConfig(early_stop_metric="filler_accuracy", early_stop_tolerance=0.01)
    assert early_stop_reached(tol_cfg, 0.995, 1.0)
    assert not early_stop_reached(tol_cfg, 0.98, 1.0)

    # cot_result_nll defaults to the measured floor and stops from above.
    nll_cfg = TrainConfig(early_stop_metric="cot_result_nll", early_stop_tolerance=0.01)
    assert resolve_early_stop_target(nll_cfg, {"cot_result_nll_floor": 0.9288}) == 0.9288
    assert early_stop_reached(nll_cfg, 0.9299, 0.9288)
    assert not early_stop_reached(nll_cfg, 2.285, 0.9288)
    # No CoT arm means no floor, so the run cannot early-stop on NLL.
    assert resolve_early_stop_target(nll_cfg, None) is None
    assert not early_stop_reached(nll_cfg, 0.9299, None)

    off_cfg = TrainConfig()
    assert resolve_early_stop_target(off_cfg, None) is None
    assert not early_stop_reached(off_cfg, 1.0, 1.0)


def test_early_stop_configuration_rejects_ambiguous_or_nonfinite_values():
    with pytest.raises(ValueError, match="early_stop_target requires"):
        TrainConfig(early_stop_target=0.9)

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="early_stop_target must be finite"):
            TrainConfig(
                early_stop_metric="filler_accuracy",
                early_stop_target=value,
            )
        with pytest.raises(ValueError, match="early_stop_tolerance must be finite"):
            TrainConfig(
                early_stop_metric="filler_accuracy",
                early_stop_tolerance=value,
            )


def test_early_stopping_does_not_change_fixed_budget_run_id():
    from exp0.evaluate import canonical_run_config

    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    baseline = canonical_run_config(model_cfg, train_cfg, task_cfg, 9999, 100, [42])
    assert "early_stop_metric" not in baseline["training_protocol"]

    enabled = replace(train_cfg, early_stop_metric="filler_accuracy")
    with_stop = canonical_run_config(model_cfg, enabled, task_cfg, 9999, 100, [42])
    assert with_stop["training_protocol"]["early_stop_metric"] == "filler_accuracy"
    assert with_stop != baseline


def test_early_stop_streak_rebuilds_from_history():
    from exp0.config import TrainConfig
    from exp0.train import _early_stop_streak

    cfg = TrainConfig(early_stop_metric="filler_accuracy")
    assert _early_stop_streak(cfg, [0.5, 1.0, 1.0], []) == 2
    assert _early_stop_streak(cfg, [1.0, 1.0, 0.5], []) == 0
    assert _early_stop_streak(cfg, [], []) == 0
    assert _early_stop_streak(TrainConfig(), [1.0, 1.0], []) == 0

    nll_cfg = TrainConfig(early_stop_metric="cot_result_nll")
    diagnostics = [
        {"cot_result_nll": 2.0, "cot_result_nll_floor": 0.9},
        {"cot_result_nll": 0.9, "cot_result_nll_floor": 0.9},
    ]
    assert _early_stop_streak(nll_cfg, [0.0, 0.0], diagnostics) == 1
    # No CoT arm means no measured floor, so no epoch can qualify.
    assert _early_stop_streak(nll_cfg, [0.0, 0.0], []) == 0


def test_train_model_stops_early_and_reports_the_shortfall():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    # target 0.0 is met by any accuracy, so the first epoch qualifies.
    train_cfg = replace(
        train_cfg,
        epochs=3,
        early_stop_metric="filler_accuracy",
        early_stop_target=0.0,
    )

    train_instances = _generate_instances(task_cfg, seed=400)
    val_instances = _generate_instances(task_cfg, seed=401)
    train_ds = Task3SumDataset(
        train_instances,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )

    _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)

    assert history["epochs_trained"] == 1
    assert history["epochs_requested"] == 3
    assert len(history["epoch_filler_accuracies"]) == 1
    early_stopping = history["early_stopping"]
    assert early_stopping["criterion_reached"] is True
    assert early_stopping["criterion_reached_after_epoch"] == 1
    assert early_stopping["triggered"] is True
    assert early_stopping["stopped_after_epoch"] == 1
    assert early_stopping["target"] == 0.0

    # Patience 2 needs a second qualifying epoch before it stops.
    patient_cfg = replace(train_cfg, early_stop_patience=2)
    _, patient_history = train_model(
        model_cfg, patient_cfg, task_cfg, train_ds, val_ds
    )
    patient_stop = patient_history["early_stopping"]
    assert patient_history["epochs_trained"] == 2
    assert patient_stop["criterion_reached_after_epoch"] == 2
    assert patient_stop["stopped_after_epoch"] == 2


def test_criterion_reached_on_final_epoch_remains_fixed_budget():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    train_cfg = replace(
        train_cfg,
        epochs=1,
        early_stop_metric="filler_accuracy",
        early_stop_target=0.0,
    )
    train_instances = _generate_instances(task_cfg, seed=600)
    val_instances = _generate_instances(task_cfg, seed=601)
    train_ds = Task3SumDataset(
        train_instances,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )

    _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)
    early_stopping = history["early_stopping"]
    assert history["epochs_trained"] == history["epochs_requested"] == 1
    assert early_stopping["criterion_reached"] is True
    assert early_stopping["criterion_reached_after_epoch"] == 1
    assert early_stopping["triggered"] is False
    assert early_stopping["stopped_after_epoch"] is None

    history["seed"] = 42
    report = compile_experiment_report(
        model_cfg,
        train_cfg,
        task_cfg,
        [history],
        majority_class_baseline=0.5,
        realized_mixture_counts=dict(train_ds.realized_counts),
        eval_seed=9999,
        val_samples=len(val_ds),
    )
    assert report["metrics"]["fixed_budget_run"] is True


def test_fixed_budget_run_reports_no_early_stop():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    train_instances = _generate_instances(task_cfg, seed=500)
    val_instances = _generate_instances(task_cfg, seed=501)
    train_ds = Task3SumDataset(
        train_instances,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )

    _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)

    assert history["epochs_trained"] == train_cfg.epochs
    assert history["epochs_requested"] == train_cfg.epochs
    assert history["early_stopping"]["enabled"] is False
    assert history["early_stopping"]["criterion_reached"] is False
    assert history["early_stopping"]["triggered"] is False


def _filler_datasets(task_cfg, num_filler, seed=600):
    """Train/validation datasets sharing a vocabulary, at a given filler budget."""
    train_instances = _generate_instances(task_cfg, seed=seed)
    val_instances = _generate_instances(task_cfg, seed=seed + 1)
    train_ds = Task3SumDataset(
        train_instances,
        num_filler=num_filler,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        num_filler=num_filler,
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    return train_ds, val_ds


def test_immediate_protocol_reports_requested_and_effective():
    """N=0 silently trains 5x the epochs; the report must show both numbers."""
    from dataclasses import replace as dc_replace

    from exp0.train import (
        IMMEDIATE_PROTOCOL_EPOCH_MULTIPLIER,
        IMMEDIATE_PROTOCOL_GRAD_CLIP,
        IMMEDIATE_PROTOCOL_WEIGHT_DECAY,
    )

    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    task_cfg = dc_replace(task_cfg, num_filler=0)
    train_cfg = dc_replace(train_cfg, epochs=1)
    train_ds, val_ds = _filler_datasets(task_cfg, num_filler=0)

    _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)

    expected = train_cfg.epochs * IMMEDIATE_PROTOCOL_EPOCH_MULTIPLIER
    assert history["epochs_requested"] == train_cfg.epochs
    assert history["epochs_effective"] == expected
    assert history["epochs_trained"] == expected
    assert len(history["epoch_filler_accuracies"]) == expected

    override = history["immediate_protocol"]
    assert override["applied"] is True
    assert override["trigger"] == "num_filler == 0"
    assert override["epochs_requested"] == train_cfg.epochs
    assert override["epochs_effective"] == expected
    assert override["weight_decay_requested"] == train_cfg.weight_decay
    assert override["weight_decay_effective"] == IMMEDIATE_PROTOCOL_WEIGHT_DECAY
    assert override["grad_clip_requested"] == train_cfg.grad_clip
    assert override["grad_clip_effective"] == IMMEDIATE_PROTOCOL_GRAD_CLIP
    # The effective values are what the optimizer actually used.
    assert history["weight_decay"] == IMMEDIATE_PROTOCOL_WEIGHT_DECAY
    assert history["grad_clip"] == IMMEDIATE_PROTOCOL_GRAD_CLIP


def test_filler_run_reports_no_immediate_protocol_override():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    train_ds, val_ds = _filler_datasets(task_cfg, num_filler=task_cfg.num_filler)

    _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)

    override = history["immediate_protocol"]
    assert override["applied"] is False
    assert override["trigger"] is None
    assert history["epochs_requested"] == train_cfg.epochs
    assert history["epochs_effective"] == train_cfg.epochs
    assert override["weight_decay_effective"] == train_cfg.weight_decay
    assert override["grad_clip_effective"] == train_cfg.grad_clip


def test_no_immediate_protocol_trains_exactly_what_was_requested():
    """--no-immediate_protocol makes an N=0 arm compute-matched to an N>0 arm."""
    from dataclasses import replace as dc_replace

    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    task_cfg = dc_replace(task_cfg, num_filler=0)
    train_cfg = dc_replace(train_cfg, epochs=1, immediate_protocol=False)
    train_ds, val_ds = _filler_datasets(task_cfg, num_filler=0, seed=700)

    _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)

    assert history["epochs_requested"] == 1
    assert history["epochs_effective"] == 1
    assert history["epochs_trained"] == 1
    assert history["weight_decay"] == train_cfg.weight_decay
    assert history["grad_clip"] == train_cfg.grad_clip

    override = history["immediate_protocol"]
    assert override["enabled"] is False
    assert override["applied"] is False
    assert override["trigger"] is None
    # The run still met the condition; record that it was suppressed rather
    # than looking identical to a run that never qualified.
    assert override["suppressed_trigger"] == "num_filler == 0"


def test_immediate_protocol_default_changes_no_existing_run_id():
    from dataclasses import replace as dc_replace

    from exp0.evaluate import canonical_run_config

    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    baseline = canonical_run_config(model_cfg, train_cfg, task_cfg, 9999, 100, [42])
    assert "immediate_protocol" not in baseline["training_protocol"]

    disabled = dc_replace(train_cfg, immediate_protocol=False)
    suppressed = canonical_run_config(model_cfg, disabled, task_cfg, 9999, 100, [42])
    assert suppressed["training_protocol"]["immediate_protocol"] is False
    assert suppressed != baseline
