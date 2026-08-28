from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from exp0.config import ModelConfig, TrainConfig
from exp0.train import (
    _autocast_context,
    _create_loader,
    _make_lr_scheduler,
    set_seed,
)
from exp1.dataset import PointerChaseDataset, exp1_collate_fn
from exp1.pointer_chase import ChaseSpec


def evaluate_vway_accuracy(
    model: torch.nn.Module,
    val_dataset: PointerChaseDataset,
    train_cfg: TrainConfig,
    device: torch.device,
) -> float:
    """Evaluate V-way accuracy by performing argmax over node-token logits.

    The predicted node is the one with the highest logit out of the V nodes.
    The answer is expected to be an integer corresponding to the node index.
    The prediction is made at the final position of the input sequence.
    """
    model.eval()
    correct = 0
    total = 0

    loader = _create_loader(
        val_dataset,
        train_cfg,
        device,
        shuffle=False,
    )
    loader.collate_fn = exp1_collate_fn

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_tuples"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with _autocast_context(device, train_cfg.precision):
                tuple_embeds = model._tuple_hidden(inputs)
                hidden_states = model.backbone(inputs_embeds=tuple_embeds)
                last_hidden = hidden_states[:, -1, :]
                logits = model.head(last_hidden)

                num_nodes = val_dataset.spec.num_nodes
                node_logits = logits[:, :num_nodes]
                predictions = node_logits.argmax(dim=-1)

                correct += predictions.eq(targets).sum().item()
                total += targets.shape[0]

    return correct / total if total > 0 else 0.0


def train_model(
    model: torch.nn.Module,
    train_dataset: PointerChaseDataset,
    val_dataset: PointerChaseDataset,
    spec: ChaseSpec,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    *,
    checkpoint_path: Optional[Path] = None,
    resume_from_checkpoint: Optional[Path] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Train loop specifically for the Experiment 1 pointer-chase task."""
    set_seed(train_cfg.seed)

    loader = _create_loader(train_dataset, train_cfg, device, shuffle=True)
    loader.collate_fn = exp1_collate_fn

    if train_cfg.fused_adamw and device.type == "cuda":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
            fused=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        )

    # Simplified single-seed setup
    epochs = train_cfg.epochs
    total_steps = len(loader) * epochs
    lr_scheduler = _make_lr_scheduler(optimizer, train_cfg, total_steps)
    scaler = torch.amp.GradScaler(device.type) if train_cfg.precision == "fp16" else None


    if resume_from_checkpoint:
        pass

    epoch_train_losses = []
    epoch_val_accuracies = []

    best_val_acc = 0.0
    optimizer_steps = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            inputs = batch["input_tuples"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with _autocast_context(device, train_cfg.precision):
                tuple_embeds = model._tuple_hidden(inputs)
                hidden_states = model.backbone(inputs_embeds=tuple_embeds)
                last_hidden = hidden_states[:, -1, :]
                logits = model.head(last_hidden)

                loss = F.cross_entropy(logits, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            total_loss += loss.item()
            optimizer_steps += 1

        epoch_loss = total_loss / len(loader)
        epoch_train_losses.append(epoch_loss)

        val_acc = evaluate_vway_accuracy(model, val_dataset, train_cfg, device)
        epoch_val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

    history = {
        "epoch_train_losses": epoch_train_losses,
        "epoch_val_accuracies": epoch_val_accuracies,
        "best_val_accuracy": best_val_acc,
        "epochs_trained": epochs,
    }

    return model, history
