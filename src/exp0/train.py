"""Training loop for Experiment 0 models."""

import random
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, pad_collate_fn
from exp0.models.base import InputEmbedWrapper
from exp0.models.llama import LlamaBackbone
from exp0.models.rwkv import RWKV7Backbone


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(model_cfg: ModelConfig, d_input: int) -> InputEmbedWrapper:
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
        )
    else:
        raise ValueError(f"Unknown architecture: {model_cfg.architecture}")

    model = InputEmbedWrapper(
        backbone=backbone,
        d_input=d_input,
        hidden_size=model_cfg.hidden_size,
        vocab_size=model_cfg.vocab_size,
    )
    return model


def evaluate_accuracy(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ans_true_id: int,
    ans_false_id: int,
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_tuples = batch["input_tuples"].to(device)
            targets = batch["targets"].to(device)
            has_3sum = batch["has_3sum"].to(device)

            logits = model(input_tuples, targets)  # (B, seq_len, vocab_size)

            # Final token prediction corresponds to true/false answer
            final_logits = logits[:, -1, :]  # (B, vocab_size)
            predictions = torch.argmax(final_logits, dim=-1)

            # Map predicted token ID to boolean answer
            pred_bool = (predictions == ans_true_id)
            correct += (pred_bool == has_3sum).sum().item()
            total += has_3sum.size(0)

    return correct / total if total > 0 else 0.0


def train_model(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    task_cfg: Task3SumConfig,
    train_dataset: Task3SumDataset,
    val_dataset: Task3SumDataset,
) -> Tuple[nn.Module, Dict[str, Any]]:
    set_seed(train_cfg.seed)
    device = torch.device(model_cfg.device)

    vocab = train_dataset.vocab
    ans_true_id = vocab.token2id.get("True", -1)
    ans_false_id = vocab.token2id.get("False", -1)

    d_input = task_cfg.mod * task_cfg.dimension + task_cfg.length
    model_cfg.vocab_size = max(len(vocab), model_cfg.vocab_size)
    model = create_model(model_cfg, d_input=d_input).to(device)

    # Condition-dependent hyperparameters (Gotcha check: immediate answer vs filler)
    is_immediate = (task_cfg.num_filler == 0) or (train_cfg.mixture == "immediate")
    weight_decay = 0.1 if is_immediate else train_cfg.weight_decay
    grad_clip = 0.5 if is_immediate else train_cfg.grad_clip
    epochs = train_cfg.epochs * 5 if is_immediate else train_cfg.epochs  # Asymmetric longer training for N=0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    epoch_val_accuracies = []
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            input_tuples = batch["input_tuples"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            optimizer.zero_grad()
            logits = model(input_tuples, targets)  # (B, seq_len, vocab_size)

            # Shift logits and targets for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_targets = loss_mask[:, 1:].contiguous().view(-1)

            loss = criterion(shift_logits, shift_targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        val_acc = evaluate_accuracy(model, val_loader, device, ans_true_id, ans_false_id)
        epoch_val_accuracies.append(val_acc)
        best_val_acc = max(best_val_acc, val_acc)

    history = {
        "best_val_accuracy": best_val_acc,
        "epoch_val_accuracies": epoch_val_accuracies,
        "epochs_trained": epochs,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
    }

    return model, history
