"""Training loop for Experiment 0 models."""

import random
import time
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
) -> DataLoader:
    """Create a DataLoader with consistent Experiment 0 settings."""
    return DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        collate_fn=pad_collate_fn,
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
        pin_memory=train_cfg.pin_memory and device.type == "cuda",
    )


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


def evaluate_accuracy(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ans_token_id: int,
    ans_true_id: int,
    ans_false_id: int,
) -> float:
    """Evaluate exact True/False prediction at the supervised ANS position."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_tuples = batch["input_tuples"].to(device)
            targets = batch["targets"].to(device)
            has_3sum = batch["has_3sum"].to(device)

            logits = model(input_tuples, targets)

            batch_size = targets.size(0)
            for b in range(batch_size):
                ans_positions = (targets[b] == ans_token_id).nonzero(as_tuple=True)[0]
                if len(ans_positions) != 1:
                    raise ValueError(
                        "Sequence must have exactly one ANS token. "
                        f"Found {len(ans_positions)}."
                    )
                ans_pos = ans_positions[0].item()

                pred_id = torch.argmax(logits[b, ans_pos, :]).item()
                expected_id = ans_true_id if has_3sum[b].item() else ans_false_id
                if pred_id == expected_id:
                    correct += 1
                total += 1

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

    # Experiment 0's output vocabulary is derived from the task schema rather
    # than selected independently. Resolve it on a local config so the caller's
    # ModelConfig remains immutable across seeds/runs.
    resolved_model_cfg = replace(model_cfg, vocab_size=len(vocab))

    d_input = task_cfg.mod * task_cfg.dimension + task_cfg.length
    model = create_model(resolved_model_cfg, d_input=d_input)
    initialization = initialize_model(model, resolved_model_cfg)
    model = model.to(device)

    # Condition-dependent hyperparameters are part of the documented 0A protocol.
    is_immediate = (task_cfg.num_filler == 0) or (
        train_cfg.mixture == "immediate"
    )
    weight_decay = 0.1 if is_immediate else train_cfg.weight_decay
    grad_clip = 0.5 if is_immediate else train_cfg.grad_clip
    epochs = train_cfg.epochs * 5 if is_immediate else train_cfg.epochs

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    train_loader = _create_loader(train_dataset, train_cfg, device, shuffle=True)
    filler_val_loader = _create_loader(
        filler_val_dataset,
        train_cfg,
        device,
        shuffle=False,
    )
    cot_val_loader = (
        _create_loader(cot_val_dataset, train_cfg, device, shuffle=False)
        if cot_val_dataset is not None
        else None
    )

    epoch_train_losses = []
    epoch_filler_accuracies = []
    epoch_cot_accuracies = []
    best_filler_acc = 0.0
    best_cot_acc = 0.0

    epoch_times = []
    data_wait = 0.0
    for _epoch in range(epochs):
        t_epoch = time.perf_counter()
        model.train()
        t_last = time.perf_counter()

        batch_losses = []
        for batch in train_loader:
            data_wait += time.perf_counter() - t_last
            input_tuples = batch["input_tuples"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            optimizer.zero_grad()
            logits = model(input_tuples, targets)

            shift_logits = logits[:, :-1, :].contiguous().view(
                -1,
                logits.size(-1),
            )
            shift_targets = loss_mask[:, 1:].contiguous().view(-1)

            loss = criterion(shift_logits, shift_targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            batch_losses.append(loss.item())
            t_last = time.perf_counter()

        epoch_train_losses.append(
            sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
        )
        epoch_times.append(time.perf_counter() - t_epoch)

        filler_acc = evaluate_accuracy(
            model,
            filler_val_loader,
            device,
            ans_token_id,
            ans_true_id,
            ans_false_id,
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
            )
            epoch_cot_accuracies.append(cot_acc)
            best_cot_acc = max(best_cot_acc, cot_acc)

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
        "initialization": initialization,
    }

    if cot_val_loader is not None:
        history["epoch_cot_accuracies"] = epoch_cot_accuracies
        history["best_cot_accuracy"] = best_cot_acc

    return model, history
