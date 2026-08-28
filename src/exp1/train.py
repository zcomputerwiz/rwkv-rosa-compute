from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from exp0.checkpointing import atomic_copy
from exp0.config import ModelConfig, TrainConfig
from exp0.train import (
    _autocast_context,
    _checkpoint_progress,
    _create_loader,
    _load_training_checkpoint_into_state,
    _make_lr_scheduler,
    _save_training_checkpoint,
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

    epochs = train_cfg.epochs
    total_steps = len(loader) * epochs
    lr_scheduler = _make_lr_scheduler(optimizer, train_cfg, total_steps)
    scaler = torch.amp.GradScaler(device.type) if train_cfg.precision == "fp16" else None

    # Dummy signature to satisfy checkpoint loader strictness if needed
    signature = {"task": "exp1_pointer_chase"}

    start_epoch = 0
    optimizer_steps = 0
    best_val_acc = 0.0

    if resume_from_checkpoint:
        prog_state, init_state = _load_training_checkpoint_into_state(
            resume_from_checkpoint,
            signature=signature,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = prog_state.get("epoch", 0)
        optimizer_steps = prog_state.get("optimizer_steps", 0)
        best_val_acc = prog_state.get("completed", {}).get("best_val_acc", 0.0)

    epoch_train_losses = []
    epoch_val_accuracies = []

    for epoch in range(start_epoch, epochs):
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

        if checkpoint_path is not None:
            prog = _checkpoint_progress(
                epoch=epoch + 1,
                epoch_seed=None,
                samples_consumed_in_epoch=0,
                optimizer_steps=optimizer_steps,
                completed={"best_val_acc": best_val_acc},
                partial_epoch=None,
            )
            epoch_checkpoint = checkpoint_path / f"epoch_{epoch + 1:03d}.pt"
            _save_training_checkpoint(
                epoch_checkpoint,
                signature=signature,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                initialization={},
                progress=prog,
            )
            atomic_copy(epoch_checkpoint, checkpoint_path / "latest.pt")

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

    history = {
        "epoch_train_losses": epoch_train_losses,
        "epoch_val_accuracies": epoch_val_accuracies,
        "best_val_accuracy": best_val_acc,
        "epochs_trained": epochs - start_epoch,
    }

    return model, history
