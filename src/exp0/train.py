"""Training loop for Experiment 0 models."""

import random
import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
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
) -> DataLoader:
    """Create a DataLoader with bounded worker/prefetch memory."""
    workers = train_cfg.num_workers if num_workers is None else num_workers
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": train_cfg.batch_size,
        "shuffle": shuffle,
        "collate_fn": pad_collate_fn,
        "num_workers": workers,
        "pin_memory": train_cfg.pin_memory and device.type == "cuda",
    }
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


def evaluate_accuracy(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ans_token_id: int,
    ans_true_id: int,
    ans_false_id: int,
    precision: str = "fp32",
    non_blocking: bool = False,
) -> float:
    """Evaluate exact True/False prediction using only ANS-position logits."""
    model.eval()
    total = 0
    correct_device = torch.zeros((), dtype=torch.int64, device=device)

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
            expected = torch.where(
                has_3sum,
                torch.full_like(predictions, ans_true_id),
                torch.full_like(predictions, ans_false_id),
            )
            correct_device.add_(predictions.eq(expected).sum())
            total += targets.shape[0]

    correct = int(correct_device.item()) if total else 0
    return correct / total if total > 0 else 0.0


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


def train_model(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    task_cfg: Task3SumConfig,
    train_dataset: Task3SumDataset,
    filler_val_dataset: Task3SumDataset,
    cot_val_dataset: Optional[Task3SumDataset] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train one Experiment 0 seed and evaluate fixed filler/CoT views."""
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

    is_immediate = (task_cfg.num_filler == 0) or (train_cfg.mixture == "immediate")
    weight_decay = 0.1 if is_immediate else train_cfg.weight_decay
    grad_clip = 0.5 if is_immediate else train_cfg.grad_clip
    epochs = train_cfg.epochs * 5 if is_immediate else train_cfg.epochs

    train_loader = _create_loader(train_dataset, train_cfg, device, shuffle=True)
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
    total_optimizer_steps = epochs * len(train_loader)
    lr_scheduler = _make_lr_scheduler(
        optimizer,
        train_cfg,
        total_steps=total_optimizer_steps,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler("cuda") if train_cfg.precision == "fp16" else None

    non_blocking = _transfer_non_blocking(train_cfg, device)
    epoch_train_losses: list[float] = []
    epoch_online_train_answer_accuracies: list[float] = []
    epoch_online_train_answer_accuracies_by_format: list[Dict[str, float | None]] = []
    epoch_filler_accuracies: list[float] = []
    epoch_cot_diagnostics: list[Dict[str, Any]] = []
    epoch_end_learning_rates: list[float] = []
    best_filler_acc = 0.0
    best_online_train_answer_acc = 0.0
    best_online_train_answer_by_format: Dict[str, float | None] = {
        name: None for name in FORMAT_NAMES
    }

    epoch_times: list[float] = []
    data_wait = 0.0
    optimizer_steps = 0

    _sync_cuda(device)
    for _epoch in range(epochs):
        t_epoch = time.perf_counter()
        model.train()
        t_last = time.perf_counter()
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
            t_last = time.perf_counter()

        _sync_cuda(device)
        epoch_times.append(time.perf_counter() - t_epoch)
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

        filler_acc = evaluate_accuracy(
            model,
            filler_val_loader,
            device,
            ans_token_id,
            ans_true_id,
            ans_false_id,
            precision=train_cfg.precision,
            non_blocking=non_blocking,
        )
        epoch_filler_accuracies.append(filler_acc)
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
            )
            epoch_cot_diagnostics.append(diagnostics)

    cuda_peak_allocated = None
    cuda_peak_reserved = None
    if device.type == "cuda":
        cuda_peak_allocated = torch.cuda.max_memory_allocated(device)
        cuda_peak_reserved = torch.cuda.max_memory_reserved(device)

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
        "epoch_val_accuracies": epoch_filler_accuracies,
        "best_filler_accuracy": best_filler_acc,
        "best_val_accuracy": best_filler_acc,
        "epochs_trained": epochs,
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
        "samples_per_second": (len(train_dataset) * epochs)
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
    }

    if epoch_cot_diagnostics:
        history["epoch_cot_diagnostics"] = epoch_cot_diagnostics
        history["best_cot_diagnostics"] = _best_cot_diagnostics(
            epoch_cot_diagnostics
        )
        history["cot_diagnostic_counts"] = epoch_cot_diagnostics[-1][
            "cot_diagnostic_counts"
        ]

    return model, history
