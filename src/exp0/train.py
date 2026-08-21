"""Training loop for Experiment 0 models."""

import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from exp0.checkpointing import (
    CHECKPOINT_VERSION,
    ResumableRandomSampler,
    atomic_copy,
    atomic_torch_save,
    capture_rng_state,
    epoch_shuffle_seed,
    load_training_checkpoint,
    optimizer_state_to_device,
    restore_rng_state,
    validate_checkpoint_signature,
)
from exp0.config import (
    EARLY_STOP_METRICS,
    ModelConfig,
    Task3SumConfig,
    TrainConfig,
    drop_identity_neutral_fields,
)
from exp0.dataset import FORMAT_NAMES, Task3SumDataset, pad_collate_fn
from exp0.diagnostics import evaluate_cot_diagnostics
from exp0.models.base import InputEmbedWrapper
from exp0.models.llama import LlamaBackbone
from exp0.models.rwkv import RWKV7Backbone
from exp0.rwkv_checkpoint import load_pretrained_backbone
from exp0.sequences import get_token_labels


def _create_loader(
    dataset: Task3SumDataset,
    train_cfg: TrainConfig,
    device: torch.device,
    shuffle: bool = False,
    num_workers: Optional[int] = None,
    sampler=None,
) -> DataLoader:
    """Create a DataLoader with bounded worker/prefetch memory."""
    workers = train_cfg.num_workers if num_workers is None else num_workers
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": train_cfg.batch_size,
        "shuffle": shuffle if sampler is None else False,
        "collate_fn": pad_collate_fn,
        "num_workers": workers,
        "pin_memory": train_cfg.pin_memory and device.type == "cuda",
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = train_cfg.prefetch_factor
    return DataLoader(**kwargs)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_match3_target_feature_map(
    vocab,
    task_cfg: Task3SumConfig,
    *,
    compact_reduced_features: bool = True,
) -> tuple[torch.Tensor, int]:
    """Map continuation vocab ids onto the shared Match-3 input feature space."""
    d_input = task_cfg.mod * task_cfg.dimension + task_cfg.length
    labels = get_token_labels(task_cfg.length)
    label_to_position = {label: idx for idx, label in enumerate(labels)}

    mapping = torch.empty(len(vocab), dtype=torch.long)
    next_feature = d_input
    special_features: dict[str, int] = {}
    compact_special_tokens = {
        vocab.pad_token,
        ":",
        ".",
        "#",
        "ANS",
        "True",
        "False",
    }
    other_feature: int | None = None

    for token_id in range(len(vocab)):
        token = vocab.id2token[token_id]
        if token in label_to_position:
            mapping[token_id] = (
                task_cfg.mod * task_cfg.dimension + label_to_position[token]
            )
        elif len(token) == 1 and token.isdigit() and int(token) < task_cfg.mod:
            mapping[token_id] = int(token)
        elif compact_reduced_features:
            if token in compact_special_tokens:
                if token not in special_features:
                    special_features[token] = next_feature
                    next_feature += 1
                mapping[token_id] = special_features[token]
            else:
                if other_feature is None:
                    other_feature = next_feature
                    next_feature += 1
                mapping[token_id] = other_feature
        else:
            mapping[token_id] = next_feature
            next_feature += 1

    return mapping, next_feature


def _initialize_llama_positive_control(
    model: InputEmbedWrapper,
    initializer_range: float,
) -> None:
    """Match Hugging Face Llama initialization without touching input_proj."""
    with torch.no_grad():
        for module in model.backbone.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(model.head.weight, mean=0.0, std=initializer_range)


def create_model(
    model_cfg: ModelConfig,
    d_input: int,
    *,
    vocab=None,
    task_cfg: Task3SumConfig | None = None,
    compact_reduced_features: bool = True,
) -> InputEmbedWrapper:
    """Construct the configured backbone and Experiment 0 task interface."""
    if model_cfg.architecture == "llama":
        backbone = LlamaBackbone(
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_hidden_layers,
            num_heads=model_cfg.num_attention_heads,
            intermediate_size=model_cfg.intermediate_size,
            rope_theta=model_cfg.llama_rope_theta,
        )
    elif model_cfg.architecture == "rwkv":
        backbone = RWKV7Backbone(
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_hidden_layers,
            intermediate_size=model_cfg.intermediate_size,
            head_dim=model_cfg.head_dim,
            rwkv_kernel=model_cfg.rwkv_kernel,
        )
    else:
        raise ValueError(f"Unknown architecture: {model_cfg.architecture}")

    target_feature_indices = None
    input_feature_dim = None
    if vocab is not None:
        if task_cfg is None:
            raise ValueError("task_cfg is required when vocab is supplied.")
        target_feature_indices, input_feature_dim = _build_match3_target_feature_map(
            vocab,
            task_cfg,
            compact_reduced_features=compact_reduced_features,
        )

    model = InputEmbedWrapper(
        backbone=backbone,
        d_input=d_input,
        hidden_size=model_cfg.hidden_size,
        vocab_size=model_cfg.vocab_size,
        output_vocab_size=model_cfg.output_vocab_size,
        target_feature_indices=target_feature_indices,
        input_feature_dim=input_feature_dim,
    )
    if model_cfg.architecture == "llama":
        _initialize_llama_positive_control(
            model,
            initializer_range=model_cfg.llama_initializer_range,
        )
    return model


def initialize_model(
    model: InputEmbedWrapper,
    model_cfg: ModelConfig,
) -> Dict[str, Any]:
    """Apply explicit random/pretrained initialization and return provenance."""
    if model_cfg.architecture == "llama":
        if model_cfg.init_mode != "random":
            raise ValueError(
                "Experiment 0 Llama runs support only random initialization."
            )
        if model_cfg.rwkv_checkpoint is not None:
            raise ValueError("rwkv_checkpoint is only valid for architecture='rwkv'.")
        return {
            "mode": "random",
            "pretrained_scope": None,
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "llama_initializer_range": model_cfg.llama_initializer_range,
            "input_adapter_init": "torch_default_linear",
            "shared_match3_input_features": True,
        }

    if model_cfg.init_mode == "random":
        if (
            model_cfg.rwkv_checkpoint is not None
            or model_cfg.rwkv_checkpoint_sha256 is not None
        ):
            raise ValueError(
                "Random RWKV initialization must not specify a pretrained checkpoint."
            )
        return {
            "mode": "random",
            "pretrained_scope": None,
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "shared_match3_input_features": True,
        }

    provenance = load_pretrained_backbone(model.backbone, model_cfg)
    provenance["task_interface_init"] = "random"
    provenance["shared_match3_input_features"] = True
    return provenance


def _validate_precision(device: torch.device, precision: str) -> None:
    if precision == "fp32":
        return
    if device.type != "cuda":
        raise ValueError(
            f"precision={precision} currently requires a CUDA device; "
            "use fp32 for CPU runs."
        )
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not report bfloat16 support.")


def _validate_cuda_backend(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> None:
    if model_cfg.rwkv_kernel != "cuda":
        return
    if model_cfg.architecture != "rwkv":
        raise ValueError("rwkv_kernel='cuda' requires architecture='rwkv'.")
    if device.type != "cuda":
        raise ValueError("rwkv_kernel='cuda' requires a CUDA device.")
    if train_cfg.precision != "bf16":
        raise ValueError(
            "The pinned RWKV-7 CUDA recurrence is BF16-only. "
            "Use --precision bf16 with --rwkv_kernel cuda."
        )


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _transfer_non_blocking(train_cfg: TrainConfig, device: torch.device) -> bool:
    return device.type == "cuda" and train_cfg.pin_memory


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# Immediate-answer protocol constants. The published protocol trains N=0 runs for
# five times the requested epochs (5 requested epochs become 25) under a
# different weight decay and gradient clip. Named here so the substitution is
# greppable rather than three bare literals inside train_model.
IMMEDIATE_PROTOCOL_EPOCH_MULTIPLIER = 5
IMMEDIATE_PROTOCOL_WEIGHT_DECAY = 0.1
IMMEDIATE_PROTOCOL_GRAD_CLIP = 0.5


def _answer_predictions_from_loss_logits(
    loss_logits: torch.Tensor,
    targets: torch.Tensor,
    ans_token_id: int,
) -> torch.Tensor:
    """Read the next-token prediction made at each ANS position."""
    ans_mask = targets[:, :-1].eq(ans_token_id)
    ans_positions = ans_mask.to(dtype=torch.int64).argmax(dim=1)
    batch_indices = torch.arange(targets.shape[0], device=targets.device)
    return loss_logits[batch_indices, ans_positions].argmax(dim=-1)


def resolve_early_stop_target(
    train_cfg: TrainConfig,
    diagnostics: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Resolve the stop target for this epoch, or None if it is unavailable.

    An explicit ``early_stop_target`` always wins. Otherwise the theoretical
    target for the metric is used: 1.0 for validation accuracy, and the measured
    ``cot_result_nll_floor`` for result-slot NLL. The NLL floor is only known
    once CoT diagnostics have run, so a run without a CoT validation arm cannot
    early-stop on that metric and will train the full budget.
    """
    if train_cfg.early_stop_target is not None:
        return float(train_cfg.early_stop_target)
    if train_cfg.early_stop_metric == "filler_accuracy":
        return 1.0
    if train_cfg.early_stop_metric == "cot_result_nll":
        if not diagnostics:
            return None
        floor = diagnostics.get("cot_result_nll_floor")
        return float(floor) if floor is not None else None
    return None


def early_stop_reached(
    train_cfg: TrainConfig,
    value: Optional[float],
    target: Optional[float],
) -> bool:
    """True when ``value`` is at the target, or within the allowed tolerance."""
    if value is None or target is None:
        return False
    direction = EARLY_STOP_METRICS.get(train_cfg.early_stop_metric)
    if direction == "max":
        return value >= target - train_cfg.early_stop_tolerance
    if direction == "min":
        return value <= target + train_cfg.early_stop_tolerance
    return False


def _early_stop_streak(
    train_cfg: TrainConfig,
    epoch_filler_accuracies: list[float],
    epoch_cot_diagnostics: list[Dict[str, Any]],
) -> int:
    """Consecutive qualifying epochs at the end of an already-recorded history.

    Resuming from a checkpoint restores the epoch metrics but not the patience
    counter, so it is recomputed here. Without this a resumed run would demand a
    fresh full patience window and train past the point an uninterrupted run
    would have stopped.
    """
    if train_cfg.early_stop_metric == "none":
        return 0
    streak = 0
    for index in range(len(epoch_filler_accuracies) - 1, -1, -1):
        diagnostics = (
            epoch_cot_diagnostics[index]
            if index < len(epoch_cot_diagnostics)
            else None
        )
        if train_cfg.early_stop_metric == "filler_accuracy":
            observed: Optional[float] = epoch_filler_accuracies[index]
        elif diagnostics is not None:
            observed = diagnostics.get(train_cfg.early_stop_metric)
        else:
            observed = None
        target = resolve_early_stop_target(train_cfg, diagnostics)
        if not early_stop_reached(train_cfg, observed, target):
            break
        streak += 1
    return streak


def first_slot_format_is_ambiguous(train_cfg: TrainConfig) -> bool:
    """True when the first post-separator target is not format-identifiable.

    Every format shares the tuple prefix and the ``:`` separator, then diverges:
    parallel CoT emits one of a pair's two labels (so the parallel share is
    split in half across two tokens), filler emits ``.``, neutral emits ``#``,
    immediate emits ``ANS`` and serial CoT emits ``DIM``. If any single
    non-CoT format outweighs ``parallel_ratio / 2`` the argmax at that slot is
    never a CoT label, and the first pair's position metric is pinned at zero
    regardless of what the model has learned.
    """
    cot_first_slot_mass = train_cfg.parallel_ratio * 0.5
    competing_mass = max(
        train_cfg.filler_ratio,
        train_cfg.serial_ratio,
        train_cfg.immediate_ratio,
        train_cfg.neutral_ratio,
    )
    return competing_mass > cot_first_slot_mass


def evaluate_accuracy(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ans_token_id: int,
    ans_true_id: int,
    ans_false_id: int,
    precision: str = "fp32",
    non_blocking: bool = False,
    return_prediction_counts: bool = False,
    detail_sink: Dict[str, list] | None = None,
) -> float | tuple[float, Dict[str, int]]:
    """Evaluate exact True/False prediction using only ANS-position logits.

    With ``return_prediction_counts`` the predicted-class histogram is returned
    alongside the accuracy. Without it there is no way to distinguish a model
    scoring at the majority-class baseline from a model that emits a single
    constant answer, which is the difference between a weak result and a null
    one.

    ``detail_sink`` optionally collects per-example predictions and the True and
    False answer logits during this same pass, so construction-stratum
    diagnostics never require a second forward pass. The return contract is
    unchanged whether or not it is supplied.
    """
    if detail_sink is not None:
        detail_sink.setdefault("predicted_ids", [])
        detail_sink.setdefault("true_logits", [])
        detail_sink.setdefault("false_logits", [])
        detail_sink.setdefault("labels", [])
    model.eval()
    total = 0
    correct_device = torch.zeros((), dtype=torch.int64, device=device)
    predicted_true_device = torch.zeros((), dtype=torch.int64, device=device)
    predicted_false_device = torch.zeros((), dtype=torch.int64, device=device)
    label_true_device = torch.zeros((), dtype=torch.int64, device=device)

    with torch.no_grad():
        for batch in val_loader:
            targets_cpu = batch["targets"]
            ans_mask_cpu = targets_cpu.eq(ans_token_id)
            ans_counts = ans_mask_cpu.sum(dim=1)
            bad = ans_counts.ne(1)
            if torch.any(bad):
                bad_count = int(ans_counts[bad][0].item())
                raise ValueError(
                    "Sequence must have exactly one ANS token. "
                    f"Found {bad_count}."
                )
            ans_positions_cpu = ans_mask_cpu.to(dtype=torch.int64).argmax(dim=1)

            input_tuples = batch["input_tuples"].to(
                device,
                non_blocking=non_blocking,
            )
            targets = targets_cpu.to(device, non_blocking=non_blocking)
            has_3sum = batch["has_3sum"].to(
                device,
                non_blocking=non_blocking,
            )
            ans_positions = ans_positions_cpu.to(
                device,
                non_blocking=non_blocking,
            )

            with _autocast_context(device, precision):
                if hasattr(model, "answer_logits"):
                    answer_logits = model.answer_logits(
                        input_tuples,
                        targets,
                        ans_positions,
                    )
                else:
                    logits = model(input_tuples, targets)
                    batch_indices = torch.arange(
                        targets.shape[0],
                        device=device,
                    )
                    answer_logits = logits[
                        batch_indices,
                        ans_positions,
                        :,
                    ]

            predictions = answer_logits.argmax(dim=-1)
            if detail_sink is not None:
                detail_sink["predicted_ids"].extend(predictions.cpu().tolist())
                detail_sink["true_logits"].extend(
                    answer_logits[:, ans_true_id].float().cpu().tolist()
                )
                detail_sink["false_logits"].extend(
                    answer_logits[:, ans_false_id].float().cpu().tolist()
                )
                detail_sink["labels"].extend(has_3sum.cpu().tolist())
            expected = torch.where(
                has_3sum,
                torch.full_like(predictions, ans_true_id),
                torch.full_like(predictions, ans_false_id),
            )
            correct_device.add_(predictions.eq(expected).sum())
            predicted_true_device.add_(predictions.eq(ans_true_id).sum())
            predicted_false_device.add_(predictions.eq(ans_false_id).sum())
            label_true_device.add_(has_3sum.sum())
            total += targets.shape[0]

    correct = int(correct_device.item()) if total else 0
    accuracy = correct / total if total > 0 else 0.0
    if not return_prediction_counts:
        return accuracy

    predicted_true = int(predicted_true_device.item()) if total else 0
    predicted_false = int(predicted_false_device.item()) if total else 0
    counts = {
        "predicted_true": predicted_true,
        "predicted_false": predicted_false,
        "predicted_other": total - predicted_true - predicted_false,
        "label_true": int(label_true_device.item()) if total else 0,
        "label_false": (total - int(label_true_device.item())) if total else 0,
        "total": total,
        "degenerate_predictor": bool(
            total > 0
            and max(predicted_true, predicted_false, total - predicted_true - predicted_false)
            == total
        ),
    }
    return accuracy, counts


def _best_cot_diagnostics(
    epoch_diagnostics: list[Dict[str, Any]],
) -> Dict[str, Any]:
    if not epoch_diagnostics:
        return {}

    metric_names = [
        "cot_answer_given_cot_accuracy",
        "cot_pair_position_token_accuracy",
        "cot_pair_position_semantic_accuracy",
        "cot_sum_token_accuracy",
        "cot_sum_semantic_accuracy",
        "cot_match_index_accuracy",
        "cot_result_semantic_accuracy",
    ]
    best: Dict[str, Any] = {}
    for name in metric_names:
        values = [item[name] for item in epoch_diagnostics if item.get(name) is not None]
        best[name] = max(values) if values else None

    nll_values = [
        item["cot_result_nll"]
        for item in epoch_diagnostics
        if item.get("cot_result_nll") is not None
    ]
    best["cot_result_nll"] = min(nll_values) if nll_values else None
    floor_values = [
        item["cot_result_nll_floor"]
        for item in epoch_diagnostics
        if item.get("cot_result_nll_floor") is not None
    ]
    best["cot_result_nll_floor"] = floor_values[-1] if floor_values else None
    if epoch_diagnostics:
        best["cot_chance_baselines"] = epoch_diagnostics[-1].get(
            "cot_chance_baselines", {}
        )
        best["cot_per_pair"] = epoch_diagnostics[-1].get("cot_per_pair", [])
        best["cot_pair_position_semantic_ceiling"] = epoch_diagnostics[-1].get(
            "cot_pair_position_semantic_ceiling", 1.0
        )
        best["cot_first_slot_format_ambiguous"] = epoch_diagnostics[-1].get(
            "cot_first_slot_format_ambiguous", False
        )
    return best


def _make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainConfig,
    total_steps: int,
) -> LambdaLR | None:
    if train_cfg.lr_schedule == "constant" or total_steps <= 0:
        return None

    warmup_steps = max(1, int(total_steps * train_cfg.warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step <= warmup_steps:
            return step / warmup_steps
        return max(0.0, 1.0 - (step / total_steps))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _checkpoint_signature(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    task_cfg: Task3SumConfig,
    train_dataset: Task3SumDataset,
    *,
    epochs: int,
    steps_per_epoch: int,
    checkpoint_run_id: str | None,
) -> dict[str, Any]:
    model_signature = asdict(model_cfg)
    model_signature.pop("rwkv_checkpoint", None)
    return {
        "run_id": checkpoint_run_id,
        "model": model_signature,
        "training": drop_identity_neutral_fields(asdict(train_cfg)),
        "task": asdict(task_cfg),
        "train_dataset_size": len(train_dataset),
        "realized_format_counts": dict(train_dataset.realized_counts),
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
    }


def _empty_completed_state() -> dict[str, Any]:
    return {
        "epoch_train_losses": [],
        "epoch_online_train_answer_accuracies": [],
        "epoch_online_train_answer_accuracies_by_format": [],
        "epoch_filler_accuracies": [],
        "epoch_filler_prediction_counts": [],
        "epoch_cot_diagnostics": [],
        "epoch_end_learning_rates": [],
        "best_filler_acc": 0.0,
        "best_filler_prediction_counts": {},
        "best_online_train_answer_acc": 0.0,
        "best_online_train_answer_by_format": {
            name: None for name in FORMAT_NAMES
        },
        "epoch_times": [],
        "data_wait": 0.0,
        "cuda_peak_memory_allocated_bytes": None,
        "cuda_peak_memory_reserved_bytes": None,
    }


def _partial_epoch_state(
    *,
    loss_sum: torch.Tensor,
    loss_count: int,
    train_answer_correct: torch.Tensor,
    train_answer_count: int,
    train_correct_by_format: torch.Tensor,
    train_count_by_format: torch.Tensor,
    epoch_elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "loss_sum": float(loss_sum.item()),
        "loss_count": int(loss_count),
        "train_answer_correct": int(train_answer_correct.item()),
        "train_answer_count": int(train_answer_count),
        "train_correct_by_format": train_correct_by_format.cpu().tolist(),
        "train_count_by_format": train_count_by_format.cpu().tolist(),
        "epoch_elapsed_seconds": float(epoch_elapsed_seconds),
    }


def _save_training_checkpoint(
    path: str | Path,
    *,
    signature: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR | None,
    scaler,
    initialization: dict[str, Any],
    progress: dict[str, Any],
) -> Path:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "signature": signature,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "initialization": initialization,
        "progress": progress,
        "rng_state": capture_rng_state(),
    }
    return atomic_torch_save(payload, path)


def _load_training_checkpoint_into_state(
    path: str | Path,
    *,
    signature: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR | None,
    scaler,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_training_checkpoint(path)
    validate_checkpoint_signature(payload.get("signature", {}), signature)

    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    optimizer_state_to_device(optimizer, device)

    saved_scheduler = payload.get("lr_scheduler_state_dict")
    if (lr_scheduler is None) != (saved_scheduler is None):
        raise ValueError("Checkpoint LR scheduler state does not match this run.")
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(saved_scheduler)

    saved_scaler = payload.get("scaler_state_dict")
    if (scaler is None) != (saved_scaler is None):
        raise ValueError("Checkpoint AMP scaler state does not match this run.")
    if scaler is not None:
        scaler.load_state_dict(saved_scaler)

    restore_rng_state(payload["rng_state"])
    return payload["progress"], payload["initialization"]


def _checkpoint_progress(
    *,
    epoch: int,
    epoch_seed: int | None,
    samples_consumed_in_epoch: int,
    optimizer_steps: int,
    completed: dict[str, Any],
    partial_epoch: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "epoch_seed": epoch_seed,
        "samples_consumed_in_epoch": int(samples_consumed_in_epoch),
        "optimizer_steps": int(optimizer_steps),
        "completed": completed,
        "partial_epoch": partial_epoch,
    }


def train_model(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    task_cfg: Task3SumConfig,
    train_dataset: Task3SumDataset,
    filler_val_dataset: Task3SumDataset,
    cot_val_dataset: Optional[Task3SumDataset] = None,
    *,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every_steps: int = 0,
    resume_checkpoint: str | Path | None = None,
    checkpoint_run_id: str | None = None,
    collect_validation_details: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train one seed, optionally writing exact-resume checkpoints.

    Periodic checkpoints are written after completed optimizer steps. They
    include partial-epoch metric accumulators and the explicit shuffled sample
    offset, so a restart neither repeats nor skips an optimizer update.
    Epoch checkpoints are written after validation/diagnostics and therefore
    resume at the next epoch with a complete history.
    """
    if checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be non-negative.")

    set_seed(train_cfg.seed)
    device = torch.device(model_cfg.device)
    _validate_precision(device, train_cfg.precision)
    _validate_cuda_backend(model_cfg, train_cfg, device)

    vocab = train_dataset.vocab
    ans_token_id = vocab.token2id.get("ANS", -1)
    ans_true_id = vocab.token2id.get("True", -1)
    ans_false_id = vocab.token2id.get("False", -1)

    if -1 in (ans_token_id, ans_true_id, ans_false_id):
        raise ValueError(
            "Vocabulary must contain 'ANS', 'True', and 'False'. "
            f"Found: ANS={ans_token_id}, True={ans_true_id}, False={ans_false_id}"
        )

    resolved_model_cfg = replace(model_cfg, vocab_size=len(vocab))
    d_input = task_cfg.mod * task_cfg.dimension + task_cfg.length
    compact_reduced_features = (
        task_cfg.vocab_reduction
        and train_dataset.realized_counts.get("serial_cot", 0) == 0
    )
    model = create_model(
        resolved_model_cfg,
        d_input=d_input,
        vocab=vocab,
        task_cfg=task_cfg,
        compact_reduced_features=compact_reduced_features,
    )
    initialization = initialize_model(model, resolved_model_cfg)
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # The published protocol trains immediate-answer runs harder than filler/CoT
    # runs: five times the epochs, and its own weight decay and gradient clip.
    # Those substitutions happen here rather than in TrainConfig, so the config a
    # run was launched with does not describe the run that executed. Every
    # overridden value is therefore reported both ways below; see
    # IMMEDIATE_PROTOCOL_EPOCH_MULTIPLIER and the "immediate_protocol" history
    # block.
    immediate_trigger = (
        "num_filler == 0"
        if task_cfg.num_filler == 0
        else "mixture == 'immediate'"
        if train_cfg.mixture == "immediate"
        else None
    )
    is_immediate = train_cfg.immediate_protocol and immediate_trigger is not None
    weight_decay = (
        IMMEDIATE_PROTOCOL_WEIGHT_DECAY if is_immediate else train_cfg.weight_decay
    )
    grad_clip = IMMEDIATE_PROTOCOL_GRAD_CLIP if is_immediate else train_cfg.grad_clip
    epochs = (
        train_cfg.epochs * IMMEDIATE_PROTOCOL_EPOCH_MULTIPLIER
        if is_immediate
        else train_cfg.epochs
    )

    checkpoint_path = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else None
    resume_path = (
        Path(resume_checkpoint).expanduser().resolve()
        if resume_checkpoint is not None
        else None
    )
    if checkpoint_path is None and resume_path is not None:
        checkpoint_path = resume_path.parent
    checkpointing_enabled = checkpoint_path is not None

    train_sampler = (
        ResumableRandomSampler(train_dataset)
        if checkpointing_enabled
        else None
    )
    train_loader = _create_loader(
        train_dataset,
        train_cfg,
        device,
        shuffle=not checkpointing_enabled,
        sampler=train_sampler,
    )
    steps_per_epoch = math.ceil(len(train_dataset) / train_cfg.batch_size)

    filler_val_loader = _create_loader(
        filler_val_dataset,
        train_cfg,
        device,
        shuffle=False,
        num_workers=train_cfg.val_num_workers,
    )
    cot_val_loader = (
        _create_loader(
            cot_val_dataset,
            train_cfg,
            device,
            shuffle=False,
            num_workers=train_cfg.val_num_workers,
        )
        if cot_val_dataset is not None
        else None
    )

    if train_cfg.fused_adamw and device.type != "cuda":
        raise ValueError("fused_adamw requires CUDA.")
    optimizer_kwargs: Dict[str, Any] = {}
    if train_cfg.fused_adamw:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )
    total_optimizer_steps = epochs * steps_per_epoch
    lr_scheduler = _make_lr_scheduler(
        optimizer,
        train_cfg,
        total_steps=total_optimizer_steps,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler("cuda") if train_cfg.precision == "fp16" else None
    non_blocking = _transfer_non_blocking(train_cfg, device)

    signature = _checkpoint_signature(
        resolved_model_cfg,
        train_cfg,
        task_cfg,
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        checkpoint_run_id=checkpoint_run_id,
    )

    completed = _empty_completed_state()
    start_epoch = 0
    resume_epoch_seed: int | None = None
    resume_samples = 0
    optimizer_steps = 0
    resume_partial: dict[str, Any] | None = None
    resumed_from: str | None = None

    if resume_path is not None:
        progress, initialization = _load_training_checkpoint_into_state(
            resume_path,
            signature=signature,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = int(progress["epoch"])
        resume_epoch_seed = progress.get("epoch_seed")
        resume_samples = int(progress["samples_consumed_in_epoch"])
        optimizer_steps = int(progress["optimizer_steps"])
        completed = progress["completed"]
        resume_partial = progress.get("partial_epoch")
        resumed_from = str(resume_path)

        if start_epoch < 0 or start_epoch > epochs:
            raise ValueError(
                f"Checkpoint epoch {start_epoch} is outside requested epochs={epochs}."
            )
        if resume_samples < 0 or resume_samples > len(train_dataset):
            raise ValueError("Checkpoint sample offset is outside the training dataset.")
        if resume_samples and resume_epoch_seed is None:
            raise ValueError("Mid-epoch checkpoint is missing its epoch shuffle seed.")
        if resume_samples and resume_partial is None:
            raise ValueError("Mid-epoch checkpoint is missing partial epoch metrics.")
        if start_epoch == epochs and resume_samples:
            raise ValueError("Completed checkpoint must have zero in-epoch sample offset.")

        print(
            f"Resumed training checkpoint {resume_path} at epoch "
            f"{start_epoch + 1 if start_epoch < epochs else epochs}/{epochs}, "
            f"optimizer step {optimizer_steps}."
        )

    epoch_train_losses = completed["epoch_train_losses"]
    epoch_online_train_answer_accuracies = completed[
        "epoch_online_train_answer_accuracies"
    ]
    epoch_online_train_answer_accuracies_by_format = completed[
        "epoch_online_train_answer_accuracies_by_format"
    ]
    epoch_filler_accuracies = completed["epoch_filler_accuracies"]
    epoch_filler_prediction_counts = completed["epoch_filler_prediction_counts"]
    epoch_cot_diagnostics = completed["epoch_cot_diagnostics"]
    epoch_end_learning_rates = completed["epoch_end_learning_rates"]
    best_filler_acc = float(completed["best_filler_acc"])
    best_filler_prediction_counts = completed["best_filler_prediction_counts"]
    best_online_train_answer_acc = float(completed["best_online_train_answer_acc"])
    best_online_train_answer_by_format = completed[
        "best_online_train_answer_by_format"
    ]
    epoch_times = completed["epoch_times"]
    data_wait = float(completed["data_wait"])
    prior_cuda_peak_allocated = completed.get("cuda_peak_memory_allocated_bytes")
    prior_cuda_peak_reserved = completed.get("cuda_peak_memory_reserved_bytes")

    final_validation_details: Dict[str, list] | None = None
    epochs_completed = start_epoch
    early_stop_criterion_reached = False
    early_stop_epoch: Optional[int] = None
    early_stop_resolved_target: Optional[float] = None
    # Rebuilt from history so a resumed run applies the same patience rule as an
    # uninterrupted one instead of restarting the streak at zero.
    early_stop_hits = _early_stop_streak(
        train_cfg, epoch_filler_accuracies, epoch_cot_diagnostics
    )

    resume_stops_before_loop = (
        resume_path is not None
        and resume_samples == 0
        and train_cfg.early_stop_metric != "none"
        and early_stop_hits >= train_cfg.early_stop_patience
    )
    if resume_stops_before_loop:
        latest_diagnostics = (
            epoch_cot_diagnostics[-1] if epoch_cot_diagnostics else None
        )
        early_stop_resolved_target = resolve_early_stop_target(
            train_cfg, latest_diagnostics
        )
        early_stop_criterion_reached = True
        # start_epoch is the number of fully completed epochs, so it is already
        # the human-readable 1-based epoch number that reached the criterion.
        early_stop_epoch = start_epoch

    loop_start_epoch = epochs if resume_stops_before_loop else start_epoch

    _sync_cuda(device)
    for _epoch in range(loop_start_epoch, epochs):
        resuming_this_epoch = _epoch == start_epoch and resume_samples > 0
        if checkpointing_enabled:
            assert train_sampler is not None
            current_epoch_seed = (
                int(resume_epoch_seed)
                if resuming_this_epoch
                else epoch_shuffle_seed(train_cfg.seed, _epoch)
            )
            samples_consumed = resume_samples if resuming_this_epoch else 0
            train_sampler.set_state(
                epoch_seed=current_epoch_seed,
                start_index=samples_consumed,
            )
        else:
            current_epoch_seed = None
            samples_consumed = 0

        model.train()
        epoch_elapsed_base = 0.0
        if resuming_this_epoch:
            partial = resume_partial or {}
            loss_sum = torch.tensor(
                float(partial["loss_sum"]), device=device, dtype=torch.float64
            )
            loss_count = int(partial["loss_count"])
            train_answer_correct = torch.tensor(
                int(partial["train_answer_correct"]), device=device, dtype=torch.int64
            )
            train_answer_count = int(partial["train_answer_count"])
            train_correct_by_format = torch.tensor(
                partial["train_correct_by_format"], device=device, dtype=torch.int64
            )
            train_count_by_format = torch.tensor(
                partial["train_count_by_format"], device=device, dtype=torch.int64
            )
            epoch_elapsed_base = float(partial["epoch_elapsed_seconds"])
        else:
            loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            loss_count = 0
            train_answer_correct = torch.zeros((), device=device, dtype=torch.int64)
            train_answer_count = 0
            train_correct_by_format = torch.zeros(
                len(FORMAT_NAMES), device=device, dtype=torch.int64
            )
            train_count_by_format = torch.zeros(
                len(FORMAT_NAMES), device=device, dtype=torch.int64
            )

        segment_start = time.perf_counter()
        checkpoint_overhead = 0.0
        t_last = time.perf_counter()

        for batch in train_loader:
            data_wait += time.perf_counter() - t_last
            input_tuples = batch["input_tuples"].to(device, non_blocking=non_blocking)
            targets = batch["targets"].to(device, non_blocking=non_blocking)
            loss_mask = batch["loss_mask"].to(device, non_blocking=non_blocking)
            has_3sum = batch["has_3sum"].to(device, non_blocking=non_blocking)
            format_codes = batch["format_code"].to(
                device, dtype=torch.long, non_blocking=non_blocking
            )

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.precision):
                loss_logits = model.loss_logits(input_tuples, targets)
                shift_targets = loss_mask[:, 1:].reshape(-1)
                loss = criterion(
                    loss_logits.reshape(-1, loss_logits.size(-1)), shift_targets
                )

            with torch.no_grad():
                answer_predictions = _answer_predictions_from_loss_logits(
                    loss_logits, targets, ans_token_id
                )
                expected_answers = torch.where(
                    has_3sum,
                    torch.full_like(answer_predictions, ans_true_id),
                    torch.full_like(answer_predictions, ans_false_id),
                )
                answer_correct = answer_predictions.eq(expected_answers)
                train_answer_correct.add_(answer_correct.sum())
                train_answer_count += targets.shape[0]
                train_count_by_format.scatter_add_(
                    0,
                    format_codes,
                    torch.ones_like(format_codes, dtype=torch.int64),
                )
                train_correct_by_format.scatter_add_(
                    0, format_codes, answer_correct.to(torch.int64)
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer_steps += 1
            loss_sum.add_(loss.detach())
            loss_count += 1
            samples_consumed += targets.shape[0]
            t_last = time.perf_counter()

            periodic_due = (
                checkpointing_enabled
                and checkpoint_every_steps > 0
                and optimizer_steps % checkpoint_every_steps == 0
                and samples_consumed < len(train_dataset)
            )
            if periodic_due:
                assert checkpoint_path is not None
                _sync_cuda(device)
                epoch_elapsed = epoch_elapsed_base + (
                    time.perf_counter() - segment_start - checkpoint_overhead
                )
                partial_state = _partial_epoch_state(
                    loss_sum=loss_sum,
                    loss_count=loss_count,
                    train_answer_correct=train_answer_correct,
                    train_answer_count=train_answer_count,
                    train_correct_by_format=train_correct_by_format,
                    train_count_by_format=train_count_by_format,
                    epoch_elapsed_seconds=epoch_elapsed,
                )
                if device.type == "cuda":
                    completed["cuda_peak_memory_allocated_bytes"] = max(
                        int(prior_cuda_peak_allocated or 0),
                        torch.cuda.max_memory_allocated(device),
                    )
                    completed["cuda_peak_memory_reserved_bytes"] = max(
                        int(prior_cuda_peak_reserved or 0),
                        torch.cuda.max_memory_reserved(device),
                    )
                completed["data_wait"] = data_wait
                save_start = time.perf_counter()
                _save_training_checkpoint(
                    checkpoint_path / "latest.pt",
                    signature=signature,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    initialization=initialization,
                    progress=_checkpoint_progress(
                        epoch=_epoch,
                        epoch_seed=current_epoch_seed,
                        samples_consumed_in_epoch=samples_consumed,
                        optimizer_steps=optimizer_steps,
                        completed=completed,
                        partial_epoch=partial_state,
                    ),
                )
                checkpoint_overhead += time.perf_counter() - save_start
                t_last = time.perf_counter()

        _sync_cuda(device)
        epoch_elapsed = epoch_elapsed_base + (
            time.perf_counter() - segment_start - checkpoint_overhead
        )
        epoch_times.append(epoch_elapsed)
        epoch_train_losses.append((loss_sum / loss_count).item() if loss_count else 0.0)
        train_answer_acc = (
            int(train_answer_correct.item()) / train_answer_count
            if train_answer_count
            else 0.0
        )
        epoch_online_train_answer_accuracies.append(train_answer_acc)
        best_online_train_answer_acc = max(best_online_train_answer_acc, train_answer_acc)

        format_correct = train_correct_by_format.cpu().tolist()
        format_count = train_count_by_format.cpu().tolist()
        format_accuracies: Dict[str, float | None] = {}
        for index, name in enumerate(FORMAT_NAMES):
            accuracy = (
                format_correct[index] / format_count[index]
                if format_count[index]
                else None
            )
            format_accuracies[name] = accuracy
            if accuracy is not None:
                prior = best_online_train_answer_by_format[name]
                best_online_train_answer_by_format[name] = (
                    accuracy if prior is None else max(prior, accuracy)
                )
        epoch_online_train_answer_accuracies_by_format.append(format_accuracies)
        epoch_end_learning_rates.append(float(optimizer.param_groups[0]["lr"]))

        # Only the last completed epoch's details are retained; each epoch
        # overwrites the previous sink, so this costs one pass, not N.
        validation_detail_sink = {} if collect_validation_details else None
        filler_acc, filler_prediction_counts = evaluate_accuracy(
            model,
            filler_val_loader,
            device,
            ans_token_id,
            ans_true_id,
            ans_false_id,
            precision=train_cfg.precision,
            non_blocking=non_blocking,
            return_prediction_counts=True,
            detail_sink=validation_detail_sink,
        )
        if validation_detail_sink is not None:
            final_validation_details = validation_detail_sink
        epoch_filler_accuracies.append(filler_acc)
        epoch_filler_prediction_counts.append(filler_prediction_counts)
        if filler_acc >= best_filler_acc:
            best_filler_prediction_counts = filler_prediction_counts
        best_filler_acc = max(best_filler_acc, filler_acc)

        if cot_val_loader is not None:
            diagnostics = evaluate_cot_diagnostics(
                model,
                cot_val_loader,
                device,
                ans_token_id,
                ans_true_id,
                ans_false_id,
                task_length=task_cfg.length,
                task_mod=task_cfg.mod,
                precision=train_cfg.precision,
                non_blocking=non_blocking,
                first_slot_format_ambiguous=first_slot_format_is_ambiguous(train_cfg),
            )
            epoch_cot_diagnostics.append(diagnostics)

        criterion_reached_this_epoch = False
        if train_cfg.early_stop_metric != "none":
            latest_diagnostics = (
                epoch_cot_diagnostics[-1] if epoch_cot_diagnostics else None
            )
            target = resolve_early_stop_target(train_cfg, latest_diagnostics)
            # Recorded every epoch so a run that never reaches the criterion
            # still reports the target it was measured against.
            early_stop_resolved_target = target
            if train_cfg.early_stop_metric == "filler_accuracy":
                observed = filler_acc
            else:
                observed = (
                    latest_diagnostics.get(train_cfg.early_stop_metric)
                    if latest_diagnostics
                    else None
                )
            if early_stop_reached(train_cfg, observed, target):
                early_stop_hits += 1
            else:
                early_stop_hits = 0
            if early_stop_hits >= train_cfg.early_stop_patience:
                early_stop_criterion_reached = True
                early_stop_epoch = _epoch + 1
                criterion_reached_this_epoch = True

        completed.update(
            {
                "best_filler_acc": best_filler_acc,
                "best_filler_prediction_counts": best_filler_prediction_counts,
                "best_online_train_answer_acc": best_online_train_answer_acc,
                "best_online_train_answer_by_format": best_online_train_answer_by_format,
                "data_wait": data_wait,
            }
        )
        if device.type == "cuda":
            completed["cuda_peak_memory_allocated_bytes"] = max(
                int(prior_cuda_peak_allocated or 0),
                torch.cuda.max_memory_allocated(device),
            )
            completed["cuda_peak_memory_reserved_bytes"] = max(
                int(prior_cuda_peak_reserved or 0),
                torch.cuda.max_memory_reserved(device),
            )

        if checkpointing_enabled:
            assert checkpoint_path is not None
            epoch_checkpoint = checkpoint_path / f"epoch_{_epoch + 1:03d}.pt"
            _save_training_checkpoint(
                epoch_checkpoint,
                signature=signature,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                initialization=initialization,
                progress=_checkpoint_progress(
                    epoch=_epoch + 1,
                    epoch_seed=None,
                    samples_consumed_in_epoch=0,
                    optimizer_steps=optimizer_steps,
                    completed=completed,
                    partial_epoch=None,
                ),
            )
            atomic_copy(epoch_checkpoint, checkpoint_path / "latest.pt")

        resume_samples = 0
        resume_epoch_seed = None
        resume_partial = None
        epochs_completed = _epoch + 1

        if criterion_reached_this_epoch:
            break

    current_cuda_peak_allocated = None
    current_cuda_peak_reserved = None
    if device.type == "cuda":
        current_cuda_peak_allocated = torch.cuda.max_memory_allocated(device)
        current_cuda_peak_reserved = torch.cuda.max_memory_reserved(device)

    cuda_peak_allocated = (
        max(int(prior_cuda_peak_allocated or 0), int(current_cuda_peak_allocated or 0))
        if device.type == "cuda"
        else None
    )
    cuda_peak_reserved = (
        max(int(prior_cuda_peak_reserved or 0), int(current_cuda_peak_reserved or 0))
        if device.type == "cuda"
        else None
    )

    early_stopped = early_stop_criterion_reached and epochs_completed < epochs
    total_train_seconds = sum(epoch_times)
    history: Dict[str, Any] = {
        "epoch_train_losses": epoch_train_losses,
        "epoch_online_train_answer_accuracies": epoch_online_train_answer_accuracies,
        "epoch_online_train_answer_accuracies_by_format": (
            epoch_online_train_answer_accuracies_by_format
        ),
        "best_online_train_answer_accuracy": best_online_train_answer_acc,
        "best_online_train_answer_accuracy_by_format": (
            best_online_train_answer_by_format
        ),
        "epoch_filler_accuracies": epoch_filler_accuracies,
        "epoch_filler_answer_prediction_counts": epoch_filler_prediction_counts,
        "best_filler_accuracy": best_filler_acc,
        "best_filler_answer_prediction_counts": best_filler_prediction_counts,
        "epochs_trained": epochs_completed,
        "epochs_requested": train_cfg.epochs,
        "epochs_effective": epochs,
        "immediate_protocol": {
            "enabled": train_cfg.immediate_protocol,
            "applied": is_immediate,
            "trigger": immediate_trigger if is_immediate else None,
            # Set when the run met the trigger but the override was disabled,
            # so a suppressed N=0 run is distinguishable from one that never
            # qualified in the first place.
            "suppressed_trigger": (
                immediate_trigger if immediate_trigger and not is_immediate else None
            ),
            "epochs_requested": train_cfg.epochs,
            "epochs_effective": epochs,
            "weight_decay_requested": train_cfg.weight_decay,
            "weight_decay_effective": weight_decay,
            "grad_clip_requested": train_cfg.grad_clip,
            "grad_clip_effective": grad_clip,
        },
        "early_stopping": {
            "enabled": train_cfg.early_stop_metric != "none",
            "metric": train_cfg.early_stop_metric,
            "target": early_stop_resolved_target,
            "tolerance": train_cfg.early_stop_tolerance,
            "patience": train_cfg.early_stop_patience,
            "criterion_reached": early_stop_criterion_reached,
            "criterion_reached_after_epoch": early_stop_epoch,
            "triggered": early_stopped,
            "stopped_after_epoch": early_stop_epoch if early_stopped else None,
            "epochs_requested": train_cfg.epochs,
            "epochs_effective": epochs,
            "epochs_trained": epochs_completed,
        },
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "adam_betas": [train_cfg.adam_beta1, train_cfg.adam_beta2],
        "lr_schedule": train_cfg.lr_schedule,
        "warmup_fraction": train_cfg.warmup_fraction,
        "optimizer_steps": optimizer_steps,
        "epoch_end_learning_rates": epoch_end_learning_rates,
        "epoch_seconds": epoch_times,
        "total_train_seconds": total_train_seconds,
        "data_wait_seconds": data_wait,
        "data_wait_fraction": data_wait / max(total_train_seconds, 1e-9),
        "samples_per_second": (len(train_dataset) * epochs_completed)
        / max(total_train_seconds, 1e-9),
        "resolved_vocab_size": len(vocab),
        "output_vocab_size": model.output_vocab_size,
        "input_feature_dim": model.input_feature_dim,
        "compact_reduced_input_features": compact_reduced_features,
        "precision": train_cfg.precision,
        "fused_adamw": train_cfg.fused_adamw,
        "rwkv_kernel": model_cfg.rwkv_kernel,
        "loss_reporting_syncs_per_epoch": 1 if device.type == "cuda" else 0,
        "validation_result_syncs_per_pass": 1 if device.type == "cuda" else 0,
        "non_blocking_transfers": non_blocking,
        "train_dataset_storage_bytes": getattr(
            train_dataset, "packed_storage_nbytes", None
        ),
        "validation_dataset_storage_bytes": getattr(
            filler_val_dataset, "packed_storage_nbytes", None
        ),
        "cuda_peak_memory_allocated_bytes": cuda_peak_allocated,
        "cuda_peak_memory_reserved_bytes": cuda_peak_reserved,
        "initialization": initialization,
        "checkpointing": {
            "enabled": checkpointing_enabled,
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_every_steps": checkpoint_every_steps,
            "checkpoint_dir": str(checkpoint_path) if checkpoint_path else None,
            "resumed_from": resumed_from,
            "exact_mid_epoch_resume": checkpointing_enabled,
        },
    }

    if final_validation_details is not None:
        history["final_validation_details"] = final_validation_details

    if epoch_cot_diagnostics:
        history["epoch_cot_diagnostics"] = epoch_cot_diagnostics
        history["best_cot_diagnostics"] = _best_cot_diagnostics(
            epoch_cot_diagnostics
        )
        history["cot_diagnostic_counts"] = epoch_cot_diagnostics[-1][
            "cot_diagnostic_counts"
        ]

    return model, history
