from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from exp0.checkpointing import ResumableRandomSampler, atomic_copy, epoch_shuffle_seed
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
    checkpoint_every_steps: Optional[int] = None,
    max_epochs: Optional[int] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Train loop specifically for the Experiment 1 pointer-chase task."""
    set_seed(train_cfg.seed)

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

    # We must calculate total_steps carefully since batch size may vary the length of loader.
    # We will build a dummy loader just to get the length.
    dummy_loader = _create_loader(train_dataset, train_cfg, device, shuffle=True)
    batches_per_epoch = len(dummy_loader)
    total_steps = batches_per_epoch * epochs
    lr_scheduler = _make_lr_scheduler(optimizer, train_cfg, total_steps)
    scaler = torch.amp.GradScaler(device.type) if train_cfg.precision == "fp16" else None

    # Dummy signature to satisfy checkpoint loader strictness if needed
    signature = {"task": "exp1_pointer_chase"}

    start_epoch = 0
    optimizer_steps = 0
    best_val_acc = 0.0
    resume_samples = 0
    resume_epoch_seed = None

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

        # If it was a mid-epoch checkpoint, prog_state has partial_epoch info
        if "partial_epoch" in prog_state and prog_state["partial_epoch"] is not None:
            partial = prog_state["partial_epoch"]
            start_epoch = partial["epoch"]
            resume_samples = partial["samples_consumed"]
            resume_epoch_seed = partial["epoch_seed"]
            optimizer_steps = prog_state.get("optimizer_steps", 0)
            comp = prog_state.get("completed", {})
            best_val_acc = comp.get("best_val_acc", 0.0)
            epoch_train_losses = comp.get("epoch_train_losses", [])
            epoch_val_accuracies = comp.get("epoch_val_accuracies", [])
        else:
            start_epoch = prog_state.get("epoch", 0)
            optimizer_steps = prog_state.get("optimizer_steps", 0)
            comp = prog_state.get("completed", {})
            best_val_acc = comp.get("best_val_acc", 0.0)
            epoch_train_losses = comp.get("epoch_train_losses", [])
            epoch_val_accuracies = comp.get("epoch_val_accuracies", [])
    else:
        epoch_train_losses = []
        epoch_val_accuracies = []

    stop_at_epoch = epochs if max_epochs is None else start_epoch + max_epochs

    for epoch in range(start_epoch, stop_at_epoch):
        model.train()
        total_loss = 0.0

        if resume_samples > 0:
            epoch_seed = resume_epoch_seed
        else:
            epoch_seed = epoch_shuffle_seed(train_cfg.seed, epoch)

        sampler = ResumableRandomSampler(
            train_dataset,
            epoch_seed=epoch_seed,
            start_index=resume_samples,
        )
        loader = _create_loader(
            train_dataset,
            train_cfg,
            device,
            sampler=sampler,
        )
        loader.collate_fn = exp1_collate_fn

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()

            inputs = batch["input_tuples"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            batch_size = targets.shape[0]

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
            resume_samples += batch_size

            if checkpoint_every_steps and optimizer_steps % checkpoint_every_steps == 0:
                if checkpoint_path is not None:
                    prog = _checkpoint_progress(
                        epoch=epoch, # In progress epoch
                        epoch_seed=epoch_seed,
                        samples_consumed_in_epoch=resume_samples,
                        optimizer_steps=optimizer_steps,
                        completed={
                            "best_val_acc": best_val_acc,
                            "epoch_train_losses": epoch_train_losses,
                            "epoch_val_accuracies": epoch_val_accuracies,
                        },
                        partial_epoch={
                            "epoch": epoch,
                            "samples_consumed": resume_samples,
                            "epoch_seed": epoch_seed,
                        },
                    )
                    step_checkpoint = checkpoint_path / f"step_{optimizer_steps:06d}.pt"
                    _save_training_checkpoint(
                        step_checkpoint,
                        signature=signature,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        initialization={},
                        progress=prog,
                    )
                    atomic_copy(step_checkpoint, checkpoint_path / "latest.pt")

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
                completed={
                    "best_val_acc": best_val_acc,
                    "epoch_train_losses": epoch_train_losses,
                    "epoch_val_accuracies": epoch_val_accuracies,
                },
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

        resume_samples = 0
        resume_epoch_seed = None

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

    history = {
        "epoch_train_losses": epoch_train_losses,
        "epoch_val_accuracies": epoch_val_accuracies,
        "best_val_accuracy": best_val_acc,
        "epochs_trained": (epochs if max_epochs is None else stop_at_epoch) - start_epoch,
    }

    return model, history
