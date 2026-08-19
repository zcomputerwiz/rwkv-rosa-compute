"""Dataset tensorization, vocabulary mapping, and compact 3SUM storage."""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
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
from exp0.task3sum import Instance3Sum, generate_instance

_FORMATS = ("parallel_cot", "filler", "serial_cot", "immediate", "neutral")
_FORMAT_TO_CODE = {name: idx for idx, name in enumerate(_FORMATS)}

COT_DIAG_NONE = 0
COT_DIAG_PAIR_POSITION = 1
COT_DIAG_SUM_RESULT = 2
COT_DIAG_MATCH_RESULT = 3


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


def build_default_vocab(
    length: int = 12,
    dimension: int = 3,
    mod: int = 10,
) -> Vocabulary:
    """Build the complete frozen Experiment 0 vocabulary."""
    vocab = Vocabulary()

    for token in (":", ".", "#", "ANS", "True", "False", "DIM", "MATCH"):
        vocab.add_token(token)

    labels = get_token_labels(length)
    for label in labels:
        vocab.add_token(label)

    for i in range(length):
        for j in range(i + 1, length):
            vocab.add_token(f"{labels[i]}{labels[j]}")
            for dim in range(dimension):
                vocab.add_token(f"{labels[i]}{labels[j]}_{dim}")
            for k in range(j + 1, length):
                for dim in range(dimension):
                    vocab.add_token(f"{labels[i]}{labels[j]}{labels[k]}_{dim}")

    for d_val in range(10**dimension):
        vocab.add_token(f"{d_val:0{dimension}d}")

    for val in range(mod):
        vocab.add_token(str(val))

    for dim in range(dimension):
        vocab.add_token(str(dim))

    vocab.freeze()
    return vocab


@dataclass
class PackedInstances:
    """Compact tensor-backed storage for a collection of 3SUM instances.

    The default 12x3 task uses 43 bytes/sample for tuple values, the Boolean
    label, and matching indices, versus a much larger graph of Python
    dataclasses/lists/tuples/integers. Torch tensor storage is also friendly to
    DataLoader multiprocessing because tensor storage can be shared between
    worker processes instead of recursively copying Python objects.
    """

    tuples: torch.Tensor
    has_3sum: torch.Tensor
    matching_indices: torch.Tensor

    def __post_init__(self):
        if self.tuples.dtype != torch.uint8 or self.tuples.ndim != 3:
            raise ValueError("Packed tuple data must be a rank-3 uint8 tensor.")
        if self.has_3sum.dtype != torch.bool or self.has_3sum.ndim != 1:
            raise ValueError("Packed labels must be a rank-1 bool tensor.")
        if self.matching_indices.dtype != torch.int16:
            raise ValueError("Packed matching indices must use int16 storage.")
        if self.matching_indices.ndim != 2 or self.matching_indices.shape[1] != 3:
            raise ValueError("Packed matching indices must have shape [N, 3].")
        count = self.tuples.shape[0]
        if (
            self.has_3sum.shape[0] != count
            or self.matching_indices.shape[0] != count
        ):
            raise ValueError("Packed instance tensors must contain the same N.")

    def __len__(self) -> int:
        return self.tuples.shape[0]

    @property
    def length(self) -> int:
        return self.tuples.shape[1]

    @property
    def dimension(self) -> int:
        return self.tuples.shape[2]

    @property
    def storage_nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.tuples, self.has_3sum, self.matching_indices)
        )

    def instance_at(self, idx: int) -> Instance3Sum:
        tuple_values = [tuple(row) for row in self.tuples[idx].tolist()]
        has_3sum = bool(self.has_3sum[idx].item())
        raw_match = self.matching_indices[idx].tolist()
        matching_indices = None
        if raw_match[0] >= 0:
            matching_indices = tuple(int(value) for value in raw_match)
        return Instance3Sum(
            tuples=tuple_values,
            has_3sum=has_3sum,
            matching_indices=matching_indices,
        )


def generate_packed_instances(
    num_samples: int,
    length: int = 12,
    dimension: int = 3,
    mod: int = 10,
    rng: Optional[random.Random] = None,
) -> PackedInstances:
    """Generate instances directly into compact shared tensor storage.

    Generation consumes the same RNG stream, in the same order, as repeatedly
    calling :func:`generate_instance`; only the retained representation changes.
    NumPy buffers are filled in-place and wrapped once with ``torch.from_numpy``
    to avoid one tiny Tensor allocation per generated sample.
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if length >= 32768:
        raise ValueError("Packed matching indices require length < 32768.")
    if rng is None:
        rng = random.Random()

    tuple_array = np.empty((num_samples, length, dimension), dtype=np.uint8)
    label_array = np.empty(num_samples, dtype=np.bool_)
    match_array = np.full((num_samples, 3), -1, dtype=np.int16)

    for idx in range(num_samples):
        instance = generate_instance(
            length=length,
            dimension=dimension,
            mod=mod,
            rng=rng,
        )
        tuple_array[idx] = instance.tuples
        label_array[idx] = instance.has_3sum
        if instance.matching_indices is not None:
            match_array[idx] = instance.matching_indices

    return PackedInstances(
        tuples=torch.from_numpy(tuple_array),
        has_3sum=torch.from_numpy(label_array),
        matching_indices=torch.from_numpy(match_array),
    )


def encode_input_tuples(instance: Instance3Sum, mod: int = 10) -> torch.Tensor:
    """Multi-hot float32 encoding for one 3SUM input."""
    tuple_values = torch.tensor(instance.tuples, dtype=torch.uint8)
    return encode_packed_input_tuples(tuple_values, mod=mod)


def encode_packed_input_tuples(
    tuple_values: torch.Tensor,
    mod: int = 10,
) -> torch.Tensor:
    """Encode one packed [length, dimension] tuple tensor without Python loops."""
    if tuple_values.ndim != 2:
        raise ValueError("tuple_values must have shape [length, dimension].")
    n, d = tuple_values.shape
    values = tuple_values.to(dtype=torch.long)
    d_input = mod * d + n
    encoded = torch.zeros((n, d_input), dtype=torch.float32)

    rows = torch.arange(n).unsqueeze(1).expand(n, d)
    dim_offsets = torch.arange(d).unsqueeze(0) * mod
    value_columns = values + dim_offsets
    encoded[rows.reshape(-1), value_columns.reshape(-1)] = 1.0

    positions = torch.arange(n)
    encoded[positions, mod * d + positions] = 1.0
    return encoded


def _matching_k_for_pair(
    instance: Instance3Sum,
    i: int,
    j: int,
    mod: int = 10,
) -> tuple[tuple[int, ...], Optional[int]]:
    dimension = len(instance.tuples[0])
    sum_ij = tuple(
        (instance.tuples[i][dim] + instance.tuples[j][dim]) % mod
        for dim in range(dimension)
    )
    target = tuple((-value) % mod for value in sum_ij)

    for k, value in enumerate(instance.tuples):
        if k != i and k != j and value == target:
            return sum_ij, k
    return sum_ij, None


def _parallel_cot_diagnostics(
    instance: Instance3Sum,
    vocab: Vocabulary,
    target_tokens: List[str],
    vocab_reduction: bool,
    include_separator_token: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build semantic target metadata for teacher-forced CoT generation.

    The reduced CoT deliberately randomizes which summand index and coordinate
    digit are emitted. Exact next-token accuracy therefore has an irreducible
    stochastic component. ``valid_ids`` records every token that is
    semantically correct for the sampled pair so evaluation can report both
    paper-style exact accuracy and stochasticity-aware semantic accuracy.
    """
    dimension = len(instance.tuples[0])
    max_valid = max(2, dimension)
    diag_type = torch.full(
        (len(target_tokens),),
        COT_DIAG_NONE,
        dtype=torch.int8,
    )
    valid_ids = torch.full(
        (len(target_tokens), max_valid),
        -1,
        dtype=torch.long,
    )

    labels = get_token_labels(len(instance.tuples))
    cursor = 1 if include_separator_token else 0

    for i in range(len(instance.tuples)):
        for j in range(i + 1, len(instance.tuples)):
            pair_pos = cursor
            result_pos = cursor + 1
            if result_pos >= len(target_tokens):
                raise ValueError("Parallel CoT diagnostic layout exceeds target length.")

            diag_type[pair_pos] = COT_DIAG_PAIR_POSITION
            if vocab_reduction:
                valid_ids[pair_pos, 0] = vocab.token2id[labels[i]]
                valid_ids[pair_pos, 1] = vocab.token2id[labels[j]]
            else:
                valid_ids[pair_pos, 0] = vocab.token2id[f"{labels[i]}{labels[j]}"]

            sum_ij, matching_k = _matching_k_for_pair(instance, i, j)
            if matching_k is not None:
                diag_type[result_pos] = COT_DIAG_MATCH_RESULT
                valid_ids[result_pos, 0] = vocab.token2id[labels[matching_k]]
            else:
                diag_type[result_pos] = COT_DIAG_SUM_RESULT
                if vocab_reduction:
                    unique_digits = list(dict.fromkeys(str(value) for value in sum_ij))
                    for offset, token in enumerate(unique_digits):
                        valid_ids[result_pos, offset] = vocab.token2id[token]
                else:
                    sum_token = "".join(str(value) for value in sum_ij)
                    valid_ids[result_pos, 0] = vocab.token2id[sum_token]

            cursor += 2

    return diag_type, valid_ids


class Task3SumDataset(Dataset):
    """3SUM dataset with compact format assignment and optional packed backing."""

    def __init__(
        self,
        instances: Sequence[Instance3Sum] | PackedInstances,
        format_type: Optional[str] = None,
        num_filler: Optional[int] = None,
        vocab: Optional[Vocabulary] = None,
        vocab_reduction: bool = True,
        include_separator_token: bool = True,
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
        self.include_separator_token = include_separator_token

        if isinstance(instances, PackedInstances):
            length = instances.length
            dimension = instances.dimension
        elif len(instances) > 0:
            length = len(instances[0].tuples)
            dimension = len(instances[0].tuples[0])
        else:
            length, dimension = 12, 3

        self.vocab = (
            build_default_vocab(length=length, dimension=dimension)
            if vocab is None
            else vocab
        )

        if format_type is not None and format_type not in _FORMAT_TO_CODE:
            raise ValueError(f"Unknown format type: {format_type}")

        self.realized_counts: Dict[str, int] = {name: 0 for name in _FORMATS}
        if format_type is not None:
            code = _FORMAT_TO_CODE[format_type]
            format_array = np.full(len(instances), code, dtype=np.uint8)
            self.realized_counts[format_type] = len(instances)
        else:
            rng = random.Random(seed)
            weights = [
                parallel_ratio,
                filler_ratio,
                serial_ratio,
                immediate_ratio,
                neutral_ratio,
            ]
            total_weight = sum(weights)
            if total_weight <= 0:
                weights = [0.5, 0.5, 0.0, 0.0, 0.0]
                total_weight = 1.0
            norm_weights = [weight / total_weight for weight in weights]

            format_array = np.empty(len(instances), dtype=np.uint8)
            for idx in range(len(instances)):
                chosen = rng.choices(_FORMATS, weights=norm_weights, k=1)[0]
                format_array[idx] = _FORMAT_TO_CODE[chosen]
                self.realized_counts[chosen] += 1

        self._format_codes = torch.from_numpy(format_array)

    def __len__(self) -> int:
        return len(self.instances)

    @property
    def assigned_formats(self) -> List[str]:
        """Compatibility view; materialized only when explicitly requested."""
        return [_FORMATS[int(code)] for code in self._format_codes.tolist()]

    @property
    def packed_storage_nbytes(self) -> Optional[int]:
        """Bytes used by compact instance and format-code tensor storage."""
        if not isinstance(self.instances, PackedInstances):
            return None
        return self.instances.storage_nbytes + (
            self._format_codes.numel() * self._format_codes.element_size()
        )

    def _instance_at(self, idx: int) -> Instance3Sum:
        if isinstance(self.instances, PackedInstances):
            return self.instances.instance_at(idx)
        return self.instances[idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        instance = self._instance_at(idx)
        fmt = _FORMATS[int(self._format_codes[idx].item())]
        item_rng = random.Random(f"{self.seed}_{idx}")

        if isinstance(self.instances, PackedInstances):
            input_embeds = encode_packed_input_tuples(self.instances.tuples[idx])
        else:
            input_embeds = encode_input_tuples(instance)

        if fmt == "parallel_cot":
            seq_str = format_a_parallel_cot(
                instance,
                vocab_reduction=self.vocab_reduction,
                rng=item_rng,
            )
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
        if ":" not in tokens:
            raise ValueError("Experiment 0 sequence is missing the ':' separator token.")
        sep_idx = tokens.index(":")
        target_tokens = tokens[
            sep_idx if self.include_separator_token else sep_idx + 1 :
        ]
        target_ids = self.vocab.encode(target_tokens)

        max_valid = max(2, len(instance.tuples[0]))
        cot_diag_type = torch.full(
            (len(target_tokens),),
            COT_DIAG_NONE,
            dtype=torch.int8,
        )
        cot_valid_ids = torch.full(
            (len(target_tokens), max_valid),
            -1,
            dtype=torch.long,
        )
        if fmt == "parallel_cot":
            cot_diag_type, cot_valid_ids = _parallel_cot_diagnostics(
                instance,
                self.vocab,
                target_tokens,
                self.vocab_reduction,
                self.include_separator_token,
            )

        return {
            "input_tuples": input_embeds,
            "targets": torch.tensor(target_ids, dtype=torch.long),
            "has_3sum": torch.tensor(instance.has_3sum, dtype=torch.bool),
            "cot_diag_type": cot_diag_type,
            "cot_valid_ids": cot_valid_ids,
            "format": fmt,
        }


def pad_collate_fn(
    batch: List[Dict[str, torch.Tensor | str]],
) -> Dict[str, torch.Tensor]:
    """Pad variable-length target sequences and create loss/diagnostic tensors."""
    input_tuples = torch.stack(
        [item["input_tuples"] for item in batch]  # type: ignore[list-item]
    )
    has_3sum = torch.stack(
        [item["has_3sum"] for item in batch]  # type: ignore[list-item]
    )
    targets_list = [item["targets"] for item in batch]
    max_len = max(target.size(0) for target in targets_list)  # type: ignore[union-attr]

    padded_targets = torch.full(
        (len(batch), max_len),
        fill_value=0,
        dtype=torch.long,
    )
    loss_masks = torch.full(
        (len(batch), max_len),
        fill_value=-100,
        dtype=torch.long,
    )
    padded_diag_types = torch.full(
        (len(batch), max_len),
        COT_DIAG_NONE,
        dtype=torch.int8,
    )
    max_valid = max(
        item["cot_valid_ids"].shape[1]  # type: ignore[union-attr]
        for item in batch
    )
    padded_valid_ids = torch.full(
        (len(batch), max_len, max_valid),
        -1,
        dtype=torch.long,
    )

    for idx, item in enumerate(batch):
        target = item["targets"]
        diag_type = item["cot_diag_type"]
        valid_ids = item["cot_valid_ids"]
        seq_len = target.size(0)  # type: ignore[union-attr]
        valid_width = valid_ids.shape[1]  # type: ignore[union-attr]
        padded_targets[idx, :seq_len] = target  # type: ignore[index]
        loss_masks[idx, :seq_len] = target  # type: ignore[index]
        padded_diag_types[idx, :seq_len] = diag_type  # type: ignore[index]
        padded_valid_ids[idx, :seq_len, :valid_width] = valid_ids  # type: ignore[index]

    return {
        "input_tuples": input_tuples,
        "targets": padded_targets,
        "loss_mask": loss_masks,
        "has_3sum": has_3sum,
        "cot_diag_type": padded_diag_types,
        "cot_valid_ids": padded_valid_ids,
    }
