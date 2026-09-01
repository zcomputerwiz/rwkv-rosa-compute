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
from exp1.qwen4_micro import Qwen4MicroConfig


def _trainable(model, workspace=None):
    """Parameters the optimizer must see.

    The workspace holds the learned routing and refinement weights. Omitting it
    here would leave M and K varying an untrained module -- every cell would
    still run, and the 2x2 would compare four frozen random workspaces.
    """
    params = list(model.parameters())
    if workspace is not None:
        params += list(workspace.parameters())
    return params


def _model_signature(model_cfg: ModelConfig | Qwen4MicroConfig) -> Dict[str, Any]:
    """Preserve legacy RWKV signatures and fully identify Qwen4-Exp runs."""
    if isinstance(model_cfg, Qwen4MicroConfig):
        return {"qwen4_exp": model_cfg.resolved()}
    return {
        "hidden_size": model_cfg.hidden_size,
        "num_hidden_layers": model_cfg.num_hidden_layers,
        "num_attention_heads": model_cfg.num_attention_heads,
        "head_dim": model_cfg.head_dim,
        "vocab_size": model_cfg.vocab_size,
        "rwkv_kernel": model_cfg.rwkv_kernel,
    }


def forward_logits(model, inputs, workspace=None):
    """The single forward path for Experiment 1, shared by training and evaluation.

    This exists because the path was duplicated. With two copies, inserting the
    workspace into one and not the other would train one architecture and
    evaluate a different one -- and every metric would still look plausible,
    because both halves run without error. The 2x2 measures a difference between
    cells, so a train/eval mismatch would not announce itself as a bug; it would
    announce itself as a result.

    ``workspace`` is applied to the final hidden state before the head. Passing
    ``None`` is the no-workspace path and is not one of the 2x2 cells: the
    baseline cell is ``M=1, K=1``, which still runs routing and refinement and
    is parameter-matched to every other cell.
    """
    tuple_embeds = model._tuple_hidden(inputs)
    hidden_states = model.backbone(inputs_embeds=tuple_embeds)
    last_hidden = hidden_states[:, -1, :]
    if workspace is not None:
        last_hidden = workspace(last_hidden)
    return model.head(last_hidden)


def evaluate_vway_accuracy(
    model: torch.nn.Module,
    val_dataset: PointerChaseDataset,
    train_cfg: TrainConfig,
    device: torch.device,
    workspace: Optional[torch.nn.Module] = None,
) -> float:
    """Evaluate V-way accuracy by performing argmax over node-token logits.

    The predicted node is the one with the highest logit out of the V nodes.
    The answer is expected to be an integer corresponding to the node index.
    The prediction is made at the final position of the input sequence.
    """
    model.eval()
    if workspace is not None:
        workspace.to(device).eval()
    correct = torch.zeros((), device=device, dtype=torch.long)
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
                logits = forward_logits(model, inputs, workspace)

                num_nodes = val_dataset.spec.num_nodes
                node_logits = logits[:, :num_nodes]
                predictions = node_logits.argmax(dim=-1)

                # Counted on the device for the same reason the training loop
                # accumulates its loss there: reading back per batch is a host
                # sync. Worth 1.18x of the evaluation pass once the backbone is
                # compiled, and nothing at all before that.
                correct += predictions.eq(targets).sum()
                total += targets.shape[0]

    # Divided in Python, not on the device. A tensor divide would be float32 and
    # would change the stored number: at correct=1, total=20000 that is
    # 0.00004999999873689376 against 0.00005, which rounds the other way at four
    # decimals. The count is exact in int64, so one readback and an integer
    # divide reproduce the original float64 result exactly.
    return correct.item() / total if total > 0 else 0.0


def train_model(
    model: torch.nn.Module,
    train_dataset: PointerChaseDataset,
    val_dataset: PointerChaseDataset,
    spec: ChaseSpec,
    model_cfg: ModelConfig | Qwen4MicroConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    *,
    workspace: Optional[torch.nn.Module] = None,
    checkpoint_path: Optional[Path] = None,
    resume_from_checkpoint: Optional[Path] = None,
    checkpoint_every_steps: Optional[int] = None,
    max_epochs: Optional[int] = None,
    # Required. There is no correct default for any of these: a default is
    # silently written into the resume signature, and two callers that both
    # omitted the same argument would produce interchangeable checkpoints.
    # That is the defect this signature exists to prevent, so it must not be
    # reachable by omission.
    depth: int,
    train_data_seed: int,
    val_data_seed: int,
    train_size: int,
    val_size: int,
    # These have genuine defaults: no silent tokens, no silent kind, the
    # generator's own queries-per-memory, and held-out evaluation.
    num_silent: int = 0,
    silent_kind: Optional[str] = None,
    queries_per_memory: int = 4,
    overfit_train_as_val: bool = False,
    # Not defaulted-and-forgotten like the fields above: whether the backbone is
    # compiled is read off the model, and this only names which backend, which
    # cannot be recovered by introspection. Passing it while the model is eager,
    # or compiling without passing it, is an error rather than a silent gap.
    compile_backend: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Train loop specifically for the Experiment 1 pointer-chase task."""
    set_seed(train_cfg.seed)

    if train_cfg.fused_adamw and device.type == "cuda":
        optimizer = torch.optim.AdamW(
            _trainable(model, workspace),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
            fused=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            _trainable(model, workspace),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        )

    if workspace is not None:
        workspace.to(device)

    checkpoint_model = model
    workspace_signature = None
    if workspace is not None:
        checkpoint_model = torch.nn.ModuleDict({"model": model, "workspace": workspace})
        workspace_signature = {
            "num_slots": workspace.num_slots,
            "num_steps": workspace.num_steps,
            "m_max": workspace.m_max,
        }

    # Simplified single-seed setup
    epochs = train_cfg.epochs

    # We must calculate total_steps carefully since batch size may vary the length of loader.
    # We will build a dummy loader just to get the length.
    dummy_loader = _create_loader(train_dataset, train_cfg, device, shuffle=True)
    batches_per_epoch = len(dummy_loader)
    total_steps = batches_per_epoch * epochs
    lr_scheduler = _make_lr_scheduler(optimizer, train_cfg, total_steps)
    scaler = torch.amp.GradScaler(device.type) if train_cfg.precision == "fp16" else None

    # The device NAME is recorded as provenance but not compared. rwkv_kernel and
    # precision already capture the parts of the device that change the
    # computation, and gating on the name would block legitimate recovery onto
    # another card.
    # out_dir and checkpoint_path are not compared. They do not enter the
    # computation.
    # Read off the model, not taken on trust: torch.compile wraps the module and
    # exposes the original as _orig_mod. Crossing the eager/compiled boundary on
    # a resume changes both the trajectory -- the inductor backend drifts from
    # eager by ~1e-3 over 15 epochs -- and the state_dict key layout, which
    # would otherwise surface as a confusing key mismatch instead of a signature
    # rejection. The backend name is not introspectable, so it is passed.
    #
    # This checks the backbone only, which is the one place the runner compiles.
    # A caller that compiled something else would not be caught here.
    backbone_compiled = hasattr(getattr(model, "backbone", None), "_orig_mod")
    if backbone_compiled and compile_backend is None:
        raise ValueError(
            "model.backbone is compiled but compile_backend was not passed; "
            "the resume signature cannot tell inductor from cudagraphs without "
            "it, and they do not produce the same trajectory"
        )
    if compile_backend is not None and not backbone_compiled:
        raise ValueError(
            f"compile_backend={compile_backend!r} was passed but model.backbone "
            f"is not compiled"
        )

    # Checked here as well as in the runner, because the runner is not the only
    # caller. cudagraphs replays into fixed-size buffers, so a ragged final
    # batch raises partway through the first evaluation -- after training has
    # run. Failing before the first step turns a wasted run into a message.
    if compile_backend == "cudagraphs":
        ragged = [f"{len(ds)} {name} instances"
                  for name, ds in (("train", train_dataset),
                                   ("val", val_dataset))
                  if len(ds) % train_cfg.batch_size]
        if ragged:
            raise ValueError(
                f"compile_backend='cudagraphs' requires every batch to have the "
                f"same shape, but {' and '.join(ragged)} are not multiples of "
                f"batch_size={train_cfg.batch_size}. Align the banks, or use "
                f"the inductor backend, which tolerates a ragged batch."
            )

    signature = {
        "task": "exp1_pointer_chase",
        "workspace": workspace_signature,
        "compile_backend": compile_backend,
        "depth": depth,
        "num_nodes": spec.num_nodes,
        "num_maps": spec.num_maps,
        "max_depth": spec.max_depth,
        "num_silent": num_silent,
        "silent_kind": silent_kind,
        "queries_per_memory": queries_per_memory,
        "train_data_seed": train_data_seed,
        "val_data_seed": val_data_seed,
        "train_size": train_size,
        "val_size": val_size,
        "overfit_train_as_val": overfit_train_as_val,
        "model_seed": train_cfg.seed,
        "batch_size": train_cfg.batch_size,
        "precision": train_cfg.precision,
        "epochs": train_cfg.epochs,
    }
    signature.update(_model_signature(model_cfg))

    start_epoch = 0
    optimizer_steps = 0
    best_val_acc = 0.0
    resume_samples = 0
    resume_epoch_seed = None
    # The running loss of a partially completed epoch. Without it a resumed
    # epoch reports the mean of only the batches that ran after the resume,
    # which is silently wrong rather than obviously wrong.
    resume_loss_sum = 0.0
    resume_batches = 0

    if resume_from_checkpoint:
        prog_state, init_state = _load_training_checkpoint_into_state(
            resume_from_checkpoint,
            signature=signature,
            model=checkpoint_model,
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
            # Absent in checkpoints written before this was carried; treating a
            # missing value as zero reproduces the old behaviour for those
            # rather than failing to load them.
            resume_loss_sum = partial.get("loss_sum", 0.0)
            resume_batches = partial.get("batches_done", 0)
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
        if workspace is not None:
            workspace.train()
        # Accumulated on the device. Reading the loss back every step is a hard
        # host sync, which stops the CPU from running ahead to queue the next
        # step's kernel launches.
        #
        # float64, not float32. The original summed loss.item() in Python, which
        # is binary64; a float32 running sum drifts by roughly (n-1)*2^-24 of the
        # mean, about 5e-5 over the 312 batches of a gate epoch -- enough to move
        # the fourth decimal of a printed and stored epoch loss. The accumulator
        # is one scalar per step, so the wider dtype costs nothing measurable
        # even where float64 throughput is poor.
        total_loss = torch.zeros((), device=device, dtype=torch.float64)
        # Seeded from the partial epoch on the first pass after a resume, and
        # zero on every epoch after that.
        total_loss += resume_loss_sum
        batches_done = resume_batches
        resume_loss_sum, resume_batches = 0.0, 0

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
                logits = forward_logits(model, inputs, workspace)

                loss = F.cross_entropy(logits, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(_trainable(model, workspace), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_trainable(model, workspace), train_cfg.grad_clip)
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            total_loss += loss.detach().double()
            batches_done += 1
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
                            "loss_sum": float(total_loss.item()),
                            "batches_done": batches_done,
                        },
                    )
                    step_checkpoint = checkpoint_path / f"step_{optimizer_steps:06d}.pt"
                    _save_training_checkpoint(
                        step_checkpoint,
                        signature=signature,
                        model=checkpoint_model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        initialization={},
                        progress=prog,
                    )
                    atomic_copy(step_checkpoint, checkpoint_path / "latest.pt")

        # Divided by the batches actually accumulated, not len(loader): after a
        # mid-epoch resume the loader yields only the remainder of the epoch,
        # while total_loss covers the whole of it.
        epoch_loss = (total_loss / max(batches_done, 1)).item()  # one sync/epoch
        epoch_train_losses.append(epoch_loss)

        val_acc = evaluate_vway_accuracy(model, val_dataset, train_cfg, device, workspace)
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
                model=checkpoint_model,
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
