"""Training loop for Experiment 0 models."""
import random
import time
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
    ans_token_id: int,
    ans_true_id: int,
    ans_false_id: int,
) -> float:
    """Evaluate accuracy at the supervised ANS position predicting True/False.

    The sequence format ends with tokens `... ANS <True|False>`.
    In causal next-token prediction, the token `ANS` predicts the next token (`True`/`False`).
    We locate the index of `ans_token_id` in `targets` for each sequence in the batch
    and read `logits[b, ans_pos, :]` to evaluate the answer prediction.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_tuples = batch["input_tuples"].to(device)
            targets = batch["targets"].to(device)
            has_3sum = batch["has_3sum"].to(device)

            logits = model(input_tuples, targets)  # (B, seq_len, vocab_size)

            batch_size = targets.size(0)
            for b in range(batch_size):
                ans_positions = (targets[b] == ans_token_id).nonzero(as_tuple=True)[0]
                if len(ans_positions) == 0:
                    ans_pos = logits.size(1) - 2
                else:
                    ans_pos = ans_positions[0].item()

                pred_logits = logits[b, ans_pos, :]
                pred_id = torch.argmax(pred_logits).item()

                pred_bool = (pred_id == ans_true_id)
                if pred_bool == has_3sum[b].item():
                    correct += 1
                total += 1

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
    ans_token_id = vocab.token2id.get("ANS", -1)
    ans_true_id = vocab.token2id.get("True", -1)
    ans_false_id = vocab.token2id.get("False", -1)

    d_input = task_cfg.mod * task_cfg.dimension + task_cfg.length
    model_cfg.vocab_size = max(len(vocab), model_cfg.vocab_size)
    model = create_model(model_cfg, d_input=d_input).to(device)

    # Condition-dependent hyperparameters
    is_immediate = (task_cfg.num_filler == 0) or (train_cfg.mixture == "immediate")
    weight_decay = 0.1 if is_immediate else train_cfg.weight_decay
    grad_clip = 0.5 if is_immediate else train_cfg.grad_clip
    epochs = train_cfg.epochs * 5 if is_immediate else train_cfg.epochs

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

    epoch_times, data_wait = [], 0.0
    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        model.train()
        t_last = time.perf_counter()
        for batch in train_loader:
            data_wait += time.perf_counter() - t_last
            input_tuples = batch["input_tuples"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            optimizer.zero_grad()
            logits = model(input_tuples, targets)

            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_targets = loss_mask[:, 1:].contiguous().view(-1)

            loss = criterion(shift_logits, shift_targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            t_last = time.perf_counter()

        epoch_times.append(time.perf_counter() - t_epoch)

        val_acc = evaluate_accuracy(model, val_loader, device, ans_token_id, ans_true_id, ans_false_id)
        epoch_val_accuracies.append(val_acc)
        best_val_acc = max(best_val_acc, val_acc)

    history = {
        "best_val_accuracy": best_val_acc,
        "epoch_val_accuracies": epoch_val_accuracies,
        "epochs_trained": epochs,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "epoch_seconds": epoch_times,
        "total_train_seconds": sum(epoch_times),
        "data_wait_seconds": data_wait,
        "samples_per_second": (len(train_dataset) * epochs) / max(sum(epoch_times), 1e-9),
    }

    return model, history
