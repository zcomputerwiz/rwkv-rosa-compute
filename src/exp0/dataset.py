"""Dataset tensorization, vocabulary mapping, and compact 3SUM storage."""

import math
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
from exp0.task3sum import (
    DEFAULT_CORRUPTION_RATE,
    SOURCE_GENERATOR,
    Instance3Sum,
    generate_instance,
    matching_k_after_pair,
)

_FORMATS = ("parallel_cot", "filler", "serial_cot", "immediate", "neutral")
FORMAT_NAMES = _FORMATS
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
    """Compact tensor-backed storage for a collection of 3SUM instances."""

    tuples: torch.Tensor
    has_3sum: torch.Tensor
    matching_indices: torch.Tensor
    # Optional generation provenance for diagnostics. Off by default so the
    # compact storage contract and every existing run are unchanged; -1 encodes
    # "not recorded" in both. construction_arm is the generator branch, not the
    # realized label.
    construction_arm: Optional[torch.Tensor] = None
    corruption_count: Optional[torch.Tensor] = None

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
        for name, provenance in (
            ("construction_arm", self.construction_arm),
            ("corruption_count", self.corruption_count),
        ):
            if provenance is None:
                continue
            if provenance.dtype != torch.int8 or provenance.ndim != 1:
                raise ValueError(f"Packed {name} must be a rank-1 int8 tensor.")
            if provenance.shape[0] != count:
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
        construction_arm = None
        if self.construction_arm is not None:
            raw_arm = int(self.construction_arm[idx].item())
            construction_arm = None if raw_arm < 0 else bool(raw_arm)
        corruption_count = None
        if self.corruption_count is not None:
            raw_corruption = int(self.corruption_count[idx].item())
            corruption_count = None if raw_corruption < 0 else raw_corruption
        return Instance3Sum(
            tuples=tuple_values,
            has_3sum=has_3sum,
            matching_indices=matching_indices,
            construction_arm=construction_arm,
            corruption_count=corruption_count,
        )


def generate_packed_instances(
    num_samples: int,
    length: int = 12,
    dimension: int = 3,
    mod: int = 10,
    rng: Optional[random.Random] = None,
    *,
    generator_mode: str = SOURCE_GENERATOR,
    corruption_rate: float = DEFAULT_CORRUPTION_RATE,
    collect_provenance: bool = False,
) -> PackedInstances:
    """Generate instances directly into compact shared tensor storage.

    ``collect_provenance`` additionally records the construction arm and
    corruption count for diagnostics. It is off by default so the compact
    storage contract is unchanged; it does not touch RNG ordering either way.
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
    arm_array = np.full(num_samples, -1, dtype=np.int8)
    corruption_array = np.full(num_samples, -1, dtype=np.int8)

    for idx in range(num_samples):
        instance = generate_instance(
            length=length,
            dimension=dimension,
            mod=mod,
            rng=rng,
            generator_mode=generator_mode,
            corruption_rate=corruption_rate,
        )
        tuple_array[idx] = instance.tuples
        label_array[idx] = instance.has_3sum
        if instance.matching_indices is not None:
            match_array[idx] = instance.matching_indices
        if collect_provenance:
            if instance.construction_arm is not None:
                arm_array[idx] = int(instance.construction_arm)
            if instance.corruption_count is not None:
                corruption_array[idx] = int(instance.corruption_count)

    return PackedInstances(
        tuples=torch.from_numpy(tuple_array),
        has_3sum=torch.from_numpy(label_array),
        matching_indices=torch.from_numpy(match_array),
        construction_arm=(
            torch.from_numpy(arm_array) if collect_provenance else None
        ),
        corruption_count=(
            torch.from_numpy(corruption_array) if collect_provenance else None
        ),
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
    return matching_k_after_pair(instance.tuples, i, j, mod=mod)


def _sum_token_entropy(sum_ij: tuple[int, ...]) -> float:
    """Entropy of the source's uniformly sampled coordinate digit target."""
    counts: dict[int, int] = {}
    for value in sum_ij:
        counts[value] = counts.get(value, 0) + 1
    dimension = len(sum_ij)
    return -sum(
        (count / dimension) * math.log(count / dimension)
        for count in counts.values()
    )


def _reduced_parallel_cot_tensors(
    instance: Instance3Sum,
    vocab: Vocabulary,
    rng: random.Random,
    include_separator_token: bool,
    include_eos_target: bool,
    mod: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build reduced CoT targets and diagnostics in one pair traversal."""
    labels = get_token_labels(len(instance.tuples))
    dimension = len(instance.tuples[0])
    max_valid = max(2, dimension)
    pair_count = len(instance.tuples) * (len(instance.tuples) - 1) // 2
    prefix = 1 if include_separator_token else 0
    suffix = 3 if include_eos_target else 2
    target_len = prefix + 2 * pair_count + suffix

    target_ids = torch.empty(target_len, dtype=torch.long)
    diag_type = torch.full((target_len,), COT_DIAG_NONE, dtype=torch.int8)
    valid_ids = torch.full((target_len, max_valid), -1, dtype=torch.long)
    pair_indices = torch.full((target_len,), -1, dtype=torch.int16)
    stochastic_nll_floor = torch.zeros(target_len, dtype=torch.float32)

    cursor = 0
    if include_separator_token:
        target_ids[cursor] = vocab.token2id[":"]
        cursor += 1

    pair_index = 0
    for i in range(len(instance.tuples)):
        for j in range(i + 1, len(instance.tuples)):
            sum_ij, matching_k = _matching_k_for_pair(instance, i, j, mod=mod)

            pair_token = labels[i] if rng.random() < 0.5 else labels[j]
            pair_id = vocab.token2id[pair_token]
            target_ids[cursor] = pair_id
            diag_type[cursor] = COT_DIAG_PAIR_POSITION
            valid_ids[cursor, 0] = vocab.token2id[labels[i]]
            valid_ids[cursor, 1] = vocab.token2id[labels[j]]
            pair_indices[cursor] = pair_index
            stochastic_nll_floor[cursor] = math.log(2.0)
            cursor += 1

            if matching_k is not None:
                result_id = vocab.token2id[labels[matching_k]]
                target_ids[cursor] = result_id
                diag_type[cursor] = COT_DIAG_MATCH_RESULT
                valid_ids[cursor, 0] = result_id
            else:
                sampled_sum = str(rng.choice(sum_ij))
                target_ids[cursor] = vocab.token2id[sampled_sum]
                diag_type[cursor] = COT_DIAG_SUM_RESULT
                unique_digits = list(dict.fromkeys(str(value) for value in sum_ij))
                for offset, token in enumerate(unique_digits):
                    valid_ids[cursor, offset] = vocab.token2id[token]
                stochastic_nll_floor[cursor] = _sum_token_entropy(sum_ij)
            pair_indices[cursor] = pair_index
            cursor += 1
            pair_index += 1

    target_ids[cursor] = vocab.token2id["ANS"]
    cursor += 1
    target_ids[cursor] = vocab.token2id[str(instance.has_3sum)]
    cursor += 1
    if include_eos_target:
        target_ids[cursor] = vocab.token2id[vocab.pad_token]
        cursor += 1
    if cursor != target_len:
        raise AssertionError("Reduced parallel CoT target construction misaligned.")

    return (
        target_ids,
        diag_type,
        valid_ids,
        pair_indices,
        stochastic_nll_floor,
    )


def _parallel_cot_diagnostics(
    instance: Instance3Sum,
    vocab: Vocabulary,
    target_tokens: List[str],
    vocab_reduction: bool,
    include_separator_token: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build semantic metadata for the non-reduced compatibility path."""
    dimension = len(instance.tuples[0])
    max_valid = max(2, dimension)
    diag_type = torch.full((len(target_tokens),), COT_DIAG_NONE, dtype=torch.int8)
    valid_ids = torch.full((len(target_tokens), max_valid), -1, dtype=torch.long)
    pair_indices = torch.full((len(target_tokens),), -1, dtype=torch.int16)
    stochastic_nll_floor = torch.zeros(len(target_tokens), dtype=torch.float32)

    labels = get_token_labels(len(instance.tuples))
    cursor = 1 if include_separator_token else 0
    pair_index = 0

    for i in range(len(instance.tuples)):
        for j in range(i + 1, len(instance.tuples)):
            pair_pos = cursor
            result_pos = cursor + 1
            if result_pos >= len(target_tokens):
                raise ValueError("Parallel CoT diagnostic layout exceeds target length.")

            diag_type[pair_pos] = COT_DIAG_PAIR_POSITION
            pair_indices[pair_pos] = pair_index
            pair_indices[result_pos] = pair_index
            if vocab_reduction:
                valid_ids[pair_pos, 0] = vocab.token2id[labels[i]]
                valid_ids[pair_pos, 1] = vocab.token2id[labels[j]]
                stochastic_nll_floor[pair_pos] = math.log(2.0)
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
                    stochastic_nll_floor[result_pos] = _sum_token_entropy(sum_ij)
                else:
                    sum_token = "".join(str(value) for value in sum_ij)
                    valid_ids[result_pos, 0] = vocab.token2id[sum_token]

            cursor += 2
            pair_index += 1

    return diag_type, valid_ids, pair_indices, stochastic_nll_floor


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
        include_eos_target: bool = True,
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
        self.include_eos_target = include_eos_target
        if not include_eos_target:
            raise ValueError(
                "Experiment 0 source-fidelity dataset requires the supervised "
                "EOS target after True/False."
            )

        if isinstance(instances, PackedInstances):
            length = instances.length
            dimension = instances.dimension
        elif len(instances) > 0:
            length = len(instances[0].tuples)
            dimension = len(instances[0].tuples[0])
        else:
            length, dimension = 12, 3
        self.length = length
        self.dimension = dimension

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
        return [_FORMATS[int(code)] for code in self._format_codes.tolist()]

    @property
    def packed_storage_nbytes(self) -> Optional[int]:
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
        format_code = int(self._format_codes[idx].item())
        fmt = _FORMATS[format_code]
        item_rng = random.Random(f"{self.seed}_{idx}")

        if isinstance(self.instances, PackedInstances):
            input_embeds = encode_packed_input_tuples(self.instances.tuples[idx])
        else:
            input_embeds = encode_input_tuples(instance)

        if fmt == "parallel_cot" and self.vocab_reduction:
            (
                target_ids,
                cot_diag_type,
                cot_valid_ids,
                cot_pair_index,
                cot_stochastic_nll_floor,
            ) = _reduced_parallel_cot_tensors(
                instance,
                self.vocab,
                item_rng,
                self.include_separator_token,
                self.include_eos_target,
            )
        else:
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
            if self.include_eos_target:
                target_tokens = [*target_tokens, self.vocab.pad_token]
            target_ids = torch.tensor(
                self.vocab.encode(target_tokens),
                dtype=torch.long,
            )

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
            cot_pair_index = torch.full(
                (len(target_tokens),),
                -1,
                dtype=torch.int16,
            )
            cot_stochastic_nll_floor = torch.zeros(
                len(target_tokens),
                dtype=torch.float32,
            )
            if fmt == "parallel_cot":
                (
                    cot_diag_type,
                    cot_valid_ids,
                    cot_pair_index,
                    cot_stochastic_nll_floor,
                ) = _parallel_cot_diagnostics(
                    instance,
                    self.vocab,
                    target_tokens,
                    self.vocab_reduction,
                    self.include_separator_token,
                )

        return {
            "input_tuples": input_embeds,
            "targets": target_ids,
            "has_3sum": torch.tensor(instance.has_3sum, dtype=torch.bool),
            "cot_diag_type": cot_diag_type,
            "cot_valid_ids": cot_valid_ids,
            "cot_pair_index": cot_pair_index,
            "cot_stochastic_nll_floor": cot_stochastic_nll_floor,
            "format_code": torch.tensor(format_code, dtype=torch.int8),
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
    format_codes = torch.stack(
        [item["format_code"] for item in batch]  # type: ignore[list-item]
    )
    targets_list = [item["targets"] for item in batch]
    max_len = max(target.size(0) for target in targets_list)  # type: ignore[union-attr]

    padded_targets = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    loss_masks = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    padded_diag_types = torch.full(
        (len(batch), max_len), COT_DIAG_NONE, dtype=torch.int8
    )
    padded_pair_indices = torch.full((len(batch), max_len), -1, dtype=torch.int16)
    padded_nll_floor = torch.zeros((len(batch), max_len), dtype=torch.float32)
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
        pair_index = item["cot_pair_index"]
        nll_floor = item["cot_stochastic_nll_floor"]
        seq_len = target.size(0)  # type: ignore[union-attr]
        valid_width = valid_ids.shape[1]  # type: ignore[union-attr]
        padded_targets[idx, :seq_len] = target  # type: ignore[index]
        loss_masks[idx, :seq_len] = target  # type: ignore[index]
        padded_diag_types[idx, :seq_len] = diag_type  # type: ignore[index]
        padded_valid_ids[idx, :seq_len, :valid_width] = valid_ids  # type: ignore[index]
        padded_pair_indices[idx, :seq_len] = pair_index  # type: ignore[index]
        padded_nll_floor[idx, :seq_len] = nll_floor  # type: ignore[index]

    return {
        "input_tuples": input_tuples,
        "targets": padded_targets,
        "loss_mask": loss_masks,
        "has_3sum": has_3sum,
        "cot_diag_type": padded_diag_types,
        "cot_valid_ids": padded_valid_ids,
        "cot_pair_index": padded_pair_indices,
        "cot_stochastic_nll_floor": padded_nll_floor,
        "format_code": format_codes,
    }
