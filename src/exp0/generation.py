"""Protocol-level Match-3 generation for Experiment 0 runs."""

import random
from typing import Optional

import numpy as np
import torch

from exp0.dataset import PackedInstances
from exp0.task3sum import (
    DEFAULT_CORRUPTION_RATE,
    SOURCE_GENERATOR,
    generate_instance,
)


def generate_protocol_packed_instances(
    num_samples: int,
    length: int = 12,
    dimension: int = 3,
    mod: int = 10,
    true_rate: float = 0.5,
    rng: Optional[random.Random] = None,
    *,
    generator_mode: str = SOURCE_GENERATOR,
    corruption_rate: float = DEFAULT_CORRUPTION_RATE,
) -> PackedInstances:
    """Generate an Experiment-0 split with its class vector sampled up front.

    The published Match-3 generator samples the True/corrupted-construction
    vector before generating any tuple contents. Pre-sampling here prevents
    variable-cost rejection/resampling for one example from changing the class
    assignment of later examples. It also makes ``Task3SumConfig.true_rate`` an
    active, provenance-worthy protocol parameter rather than unused metadata.
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if not 0.0 <= true_rate <= 1.0:
        raise ValueError("true_rate must be in [0, 1].")
    if length >= 32768:
        raise ValueError("Packed matching indices require length < 32768.")
    if rng is None:
        rng = random.Random()

    requested_labels = [rng.random() < true_rate for _ in range(num_samples)]
    tuple_array = np.empty((num_samples, length, dimension), dtype=np.uint8)
    label_array = np.empty(num_samples, dtype=np.bool_)
    match_array = np.full((num_samples, 3), -1, dtype=np.int16)

    for idx, target_has_3sum in enumerate(requested_labels):
        instance = generate_instance(
            length=length,
            dimension=dimension,
            mod=mod,
            target_has_3sum=target_has_3sum,
            rng=rng,
            generator_mode=generator_mode,
            corruption_rate=corruption_rate,
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
