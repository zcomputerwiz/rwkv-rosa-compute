"""Training loop for Experiment 0 models."""

import random
import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, pad_collate_fn
from exp0.models.base import InputEmbedWrapper
from exp0.models.llama import LlamaBackbone
from exp0.models.rwkv import RWKV7Backbone
from exp0.rwkv_checkpoint import load_pretrained_backbone


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


def create_model(model_cfg: ModelConfig, d_input: int) -> InputEmbedWrapper:
    """Construct the configured backbone and Experiment 0 task interface."""
    if model_cfg.architecture == "llama":
        backbone = LlamaBackbone(
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_hidden_layers,
            num_heads=model_cfg.num_attention_heads,
            intermediate_size=model_cfg.intermediate_size,
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

    return InputEmbedWrapper(
        backbone=backbone,
        d_input=d_input,
        hidden_size=model_cfg.hidden_size,
        vocab_size=model_cfg.vocab_size,
    )


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
        }

    provenance = load_pretrained_backbone(model.backbone, model_cfg)
    provenance["task_interface_init"] = "random"
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
            # Validate the structural ANS contract on the CPU copy before moving
            # the batch. This avoids a GPU truth-value synchronization per batch.
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
            targets = targets_cpu.to(
                device,
                non_blocking=non_blocking,
            )
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
            f"Found: ANS={ans_token_id}, True={ans_true_id}, "
            f"False={ans_false_id}"
        )

    resolved_model_cfg = replace(model_cfg, vocab_size=len(vocab))
    d_input = task_cfg.mod * task_cfg.dimension + task_cfg.length
    model = create_model(resolved_model_cfg, d_input=d_input)
    initialization = initialize_model(model, resolved_model_cfg)
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    is_immediate = (task_cfg.num_filler == 0) or (
        train_cfg.mixture == "immediate"
    )
    weight_decay = 0.1 if is_immediate else train_cfg.weight_decay
    grad_clip = 0.5 if is_immediate else train_cfg.grad_clip
    epochs = train_cfg.epochs * 5 if is_immediate else train_cfg.epochs

    if train_cfg.fused_adamw and device.type != "cuda":
        raise ValueError("fused_adamw requires CUDA.")
    optimizer_kwargs: Dict[str, Any] = {}
    if train_cfg.fused_adamw:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = (
        torch.amp.GradScaler("cuda")
        if train_cfg.precision == "fp16"
        else None
    )

    train_loader = _create_loader(
        train_dataset,
        train_cfg,
        device,
        shuffle=True,
    )
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

    non_blocking = _transfer_non_blocking(train_cfg, device)
    epoch_train_losses = []
    epoch_filler_accuracies = []
    epoch_cot_accuracies = []
    best_filler_acc = 0.0
    best_cot_acc = 0.0

    epoch_times = []
    data_wait = 0.0

    # Ensure model initialization / transfers are outside the first epoch timer.
    # Thereafter validation ends with one accuracy scalar read, so the preceding
    # CUDA work is already complete before the next epoch begins.
    _sync_cuda(device)
    for _epoch in range(epochs):
        t_epoch = time.perf_counter()
        model.train()
        t_last = time.perf_counter()
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        loss_count = 0

        for batch in train_loader:
            data_wait += time.perf_counter() - t_last
            input_tuples = batch["input_tuples"].to(
                device,
                non_blocking=non_blocking,
            )
            targets = batch["targets"].to(
                device,
                non_blocking=non_blocking,
            )
            loss_mask = batch["loss_mask"].to(
                device,
                non_blocking=non_blocking,
            )

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.precision):
                loss_logits = model.loss_logits(input_tuples, targets)
                shift_targets = loss_mask[:, 1:].reshape(-1)
                loss = criterion(
                    loss_logits.reshape(-1, loss_logits.size(-1)),
                    shift_targets,
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

            # PR #9 introduced loss.item() solely to record epoch-mean training
            # loss. Detaching preserves that reporting purpose without retaining
            # autograd graphs; accumulating on-device avoids a CUDA sync per batch.
            loss_sum.add_(loss.detach())
            loss_count += 1
            t_last = time.perf_counter()

        # One synchronization at the epoch boundary preserves accurate wall-clock
        # epoch_seconds / samples_per_second after removing per-batch loss.item().
        _sync_cuda(device)
        epoch_times.append(time.perf_counter() - t_epoch)
        epoch_train_losses.append(
            (loss_sum / loss_count).item() if loss_count else 0.0
        )

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
            cot_acc = evaluate_accuracy(
                model,
                cot_val_loader,
                device,
                ans_token_id,
                ans_true_id,
                ans_false_id,
                precision=train_cfg.precision,
                non_blocking=non_blocking,
            )
            epoch_cot_accuracies.append(cot_acc)
            best_cot_acc = max(best_cot_acc, cot_acc)

    cuda_peak_allocated = None
    cuda_peak_reserved = None
    if device.type == "cuda":
        cuda_peak_allocated = torch.cuda.max_memory_allocated(device)
        cuda_peak_reserved = torch.cuda.max_memory_reserved(device)

    history: Dict[str, Any] = {
        "epoch_train_losses": epoch_train_losses,
        "epoch_filler_accuracies": epoch_filler_accuracies,
        "epoch_val_accuracies": epoch_filler_accuracies,
        "best_filler_accuracy": best_filler_acc,
        "best_val_accuracy": best_filler_acc,
        "epochs_trained": epochs,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "epoch_seconds": epoch_times,
        "total_train_seconds": sum(epoch_times),
        "data_wait_seconds": data_wait,
        "samples_per_second": (len(train_dataset) * epochs)
        / max(sum(epoch_times), 1e-9),
        "resolved_vocab_size": len(vocab),
        "precision": train_cfg.precision,
        "fused_adamw": train_cfg.fused_adamw,
        "rwkv_kernel": model_cfg.rwkv_kernel,
        "loss_reporting_syncs_per_epoch": 1 if device.type == "cuda" else 0,
        "validation_result_syncs_per_pass": 1 if device.type == "cuda" else 0,
        "non_blocking_transfers": non_blocking,
        "train_dataset_storage_bytes": getattr(
            train_dataset,
            "packed_storage_nbytes",
            None,
        ),
        "validation_dataset_storage_bytes": getattr(
            filler_val_dataset,
            "packed_storage_nbytes",
            None,
        ),
        "cuda_peak_memory_allocated_bytes": cuda_peak_allocated,
        "cuda_peak_memory_reserved_bytes": cuda_peak_reserved,
        "initialization": initialization,
    }

    if cot_val_loader is not None:
        history["epoch_cot_accuracies"] = epoch_cot_accuracies
        history["best_cot_accuracy"] = best_cot_acc

    return model, history
