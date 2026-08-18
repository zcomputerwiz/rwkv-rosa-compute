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
        self.add_token(self.pad_token)

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2id.get(t, 0) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id2token.get(i, "<UNK>") for i in ids]

    def __len__(self):
        return len(self.token2id)


def build_default_vocab(length: int = 12, dimension: int = 3, mod: int = 10) -> Vocabulary:
    """Build vocabulary covering all possible tokens across formats A-E for specified length/dimension."""
    vocab = Vocabulary()

    # Separator, filler, neutral, answer tokens
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
                    vocab.add_token(f"MATCH {labels[i]}{labels[j]}{labels[k]}_{dim}")
                    vocab.add_token(f"{labels[i]}{labels[j]}{labels[k]}_{dim}")

    # Digits / sums
    for d_val in range(10**dimension):
        d_str = f"{d_val:0{dimension}d}"
        vocab.add_token(d_str)

    for val in range(mod):
        vocab.add_token(str(val))

    # Dimension markers for serial CoT
    for dim in range(dimension):
        vocab.add_token(str(dim))

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
    """PyTorch Dataset for 3SUM experiment."""

    def __init__(
        self,
        instances: List[Instance3Sum],
        format_type: str = "parallel_cot",
        num_filler: Optional[int] = None,
        vocab: Optional[Vocabulary] = None,
        vocab_reduction: bool = True,
        seed: int = 42,
    ):
        self.instances = instances
        self.format_type = format_type
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

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        instance = self.instances[idx]

        input_embeds = encode_input_tuples(instance)

        if self.format_type == "parallel_cot":
            seq_str = format_a_parallel_cot(instance, vocab_reduction=self.vocab_reduction, rng=self.rng)
        elif self.format_type == "filler":
            seq_str = format_b_filler(instance, num_filler=self.num_filler)
        elif self.format_type == "immediate":
            seq_str = format_c_immediate(instance)
        elif self.format_type == "serial_cot":
            seq_str = format_d_serial_cot(instance)
        elif self.format_type == "neutral":
            seq_str = format_e_neutral(instance, num_filler=self.num_filler)
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")

        tokens = seq_str.split()

        sep_idx = tokens.index(":") if ":" in tokens else -1
        target_tokens = tokens[sep_idx + 1:] if sep_idx != -1 else tokens

        target_ids = []
        for t in target_tokens:
            target_ids.append(self.vocab.add_token(t))

        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        return {
            "input_tuples": input_embeds,
            "targets": target_tensor,
            "has_3sum": torch.tensor(instance.has_3sum, dtype=torch.bool),
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
