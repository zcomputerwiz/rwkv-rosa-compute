"""Dataset tensorization, vocabulary mapping, and loss masking for 3SUM sequence formats."""

import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from exp0.sequences import (
    format_a_parallel_cot,
    format_b_filler,
    format_c_immediate,
    format_d_serial_cot,
    format_e_neutral,
    get_token_labels,
)
from exp0.task3sum import Instance3Sum


class Vocabulary:
    """Vocabulary mapping string tokens to discrete integer token IDs."""

    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.pad_token = "<PAD>"
        self.is_frozen = False
        self.add_token(self.pad_token)

    def add_token(self, token: str) -> int:
        if token in self.token2id:
            return self.token2id[token]
        if self.is_frozen:
            raise KeyError(f"Vocabulary is frozen. Cannot add new token: {token}")
        idx = len(self.token2id)
        self.token2id[token] = idx
        self.id2token[idx] = token
        return idx

    def freeze(self):
        self.is_frozen = True

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2id[t] for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id2token.get(i, "<UNK>") for i in ids]

    def __len__(self):
        return len(self.token2id)


def build_default_vocab(length: int = 12, dimension: int = 3, mod: int = 10) -> Vocabulary:
    """Build complete, clean default vocabulary for specified length/dimension."""
    vocab = Vocabulary()

    # Separators, filler, neutral, answer tokens
    vocab.add_token(":")
    vocab.add_token(".")
    vocab.add_token("#")
    vocab.add_token("ANS")
    vocab.add_token("True")
    vocab.add_token("False")
    vocab.add_token("DIM")
    vocab.add_token("MATCH")

    # Labels
    labels = get_token_labels(length)
    for lbl in labels:
        vocab.add_token(lbl)

    # All pairs
    for i in range(length):
        for j in range(i + 1, length):
            vocab.add_token(f"{labels[i]}{labels[j]}")
            for dim in range(dimension):
                vocab.add_token(f"{labels[i]}{labels[j]}_{dim}")
            for k in range(j + 1, length):
                for dim in range(dimension):
                    vocab.add_token(f"{labels[i]}{labels[j]}{labels[k]}_{dim}")

    # Digits / sums
    for d_val in range(10**dimension):
        d_str = f"{d_val:0{dimension}d}"
        vocab.add_token(d_str)

    for val in range(mod):
        vocab.add_token(str(val))

    # Dimension markers
    for dim in range(dimension):
        vocab.add_token(str(dim))

    vocab.freeze()
    return vocab


def encode_input_tuples(instance: Instance3Sum, mod: int = 10) -> torch.Tensor:
    """Multi-hot binary vector encoding for input tuples."""
    n = len(instance.tuples)
    d = len(instance.tuples[0])
    d_input = mod * d + n

    encoded = torch.zeros(n, d_input, dtype=torch.float32)
    for i, tup in enumerate(instance.tuples):
        for dim, digit in enumerate(tup):
            encoded[i, dim * mod + digit] = 1.0
        encoded[i, mod * d + i] = 1.0
    return encoded


class Task3SumDataset(Dataset):
    """PyTorch Dataset for 3SUM experiment supporting training mixtures."""

    def __init__(
        self,
        instances: List[Instance3Sum],
        format_type: Optional[str] = None,  # Single format override if specified
        num_filler: Optional[int] = None,
        vocab: Optional[Vocabulary] = None,
        vocab_reduction: bool = True,
        seed: int = 42,
        parallel_ratio: float = 0.5,
        filler_ratio: float = 0.5,
        serial_ratio: float = 0.0,
        immediate_ratio: float = 0.0,
        neutral_ratio: float = 0.0,
    ):
        self.instances = instances
        self.seed = seed
        self.num_filler = num_filler
        self.vocab_reduction = vocab_reduction
        self.rng = random.Random(seed)

        if len(instances) > 0:
            length = len(instances[0].tuples)
            dimension = len(instances[0].tuples[0])
        else:
            length, dimension = 12, 3

        if vocab is None:
            self.vocab = build_default_vocab(length=length, dimension=dimension)
        else:
            self.vocab = vocab

        # Determine mixture format assignment per instance
        self.assigned_formats: List[str] = []
        self.realized_counts: Dict[str, int] = {
            "parallel_cot": 0,
            "filler": 0,
            "serial_cot": 0,
            "immediate": 0,
            "neutral": 0,
        }

        if format_type is not None:
            # Single format override
            for _ in self.instances:
                self.assigned_formats.append(format_type)
                self.realized_counts[format_type] += 1
        else:
            # Sample format per instance based on mixture ratios
            formats = ["parallel_cot", "filler", "serial_cot", "immediate", "neutral"]
            weights = [parallel_ratio, filler_ratio, serial_ratio, immediate_ratio, neutral_ratio]
            total_weight = sum(weights)
            if total_weight <= 0:
                weights = [0.5, 0.5, 0.0, 0.0, 0.0]
                total_weight = 1.0

            norm_weights = [w / total_weight for w in weights]

            for _ in self.instances:
                chosen_format = self.rng.choices(formats, weights=norm_weights, k=1)[0]
                self.assigned_formats.append(chosen_format)
                self.realized_counts[chosen_format] += 1

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        instance = self.instances[idx]
        fmt = self.assigned_formats[idx]

        # Deterministic per-item RNG
        item_rng = random.Random(f"{self.seed}_{idx}")

        input_embeds = encode_input_tuples(instance)

        if fmt == "parallel_cot":
            seq_str = format_a_parallel_cot(instance, vocab_reduction=self.vocab_reduction, rng=item_rng)
        elif fmt == "filler":
            seq_str = format_b_filler(instance, num_filler=self.num_filler)
        elif fmt == "immediate":
            seq_str = format_c_immediate(instance)
        elif fmt == "serial_cot":
            seq_str = format_d_serial_cot(instance)
        elif fmt == "neutral":
            seq_str = format_e_neutral(instance, num_filler=self.num_filler)
        else:
            raise ValueError(f"Unknown format type: {fmt}")

        tokens = seq_str.split()

        sep_idx = tokens.index(":") if ":" in tokens else -1
        target_tokens = tokens[sep_idx + 1:] if sep_idx != -1 else tokens

        # Encode tokens using frozen vocabulary
        target_ids = self.vocab.encode(target_tokens)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        return {
            "input_tuples": input_embeds,
            "targets": target_tensor,
            "has_3sum": torch.tensor(instance.has_3sum, dtype=torch.bool),
            "format": fmt,
        }


def pad_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length target sequences and create target/loss mask tensors."""
    input_tuples = torch.stack([item["input_tuples"] for item in batch])
    has_3sum = torch.stack([item["has_3sum"] for item in batch])

    targets_list = [item["targets"] for item in batch]
    max_len = max(t.size(0) for t in targets_list)

    padded_targets = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    loss_masks = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)

    for i, t in enumerate(targets_list):
        seq_len = t.size(0)
        padded_targets[i, :seq_len] = t
        loss_masks[i, :seq_len] = t

    return {
        "input_tuples": input_tuples,
        "targets": padded_targets,
        "loss_mask": loss_masks,
        "has_3sum": has_3sum,
    }
