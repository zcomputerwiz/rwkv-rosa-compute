"""3SUM task instance generation and solving logic."""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Instance3Sum:
    """A 3SUM instance consisting of n d-dimensional tuples in Z_10^d."""

    tuples: List[Tuple[int, ...]]
    has_3sum: bool
    matching_indices: Optional[Tuple[int, int, int]] = None


def check_3sum(
    tuples: List[Tuple[int, ...]],
    mod: int = 10,
) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
    """Return the first matching triple using exact legacy search ordering.

    The legacy implementation scanned ``i``, then ``j``, then every possible
    ``k``. This version preserves that ordering while replacing the innermost
    full scan with a tuple-value index, reducing the common search cost from
    O(n^3) to O(n^2) lookups plus duplicate-value checks.
    """
    n = len(tuples)
    d = len(tuples[0])

    value_to_indices: dict[Tuple[int, ...], list[int]] = {}
    for idx, value in enumerate(tuples):
        value_to_indices.setdefault(value, []).append(idx)

    for i in range(n):
        for j in range(i + 1, n):
            target = tuple(
                (-tuples[i][dim] - tuples[j][dim]) % mod
                for dim in range(d)
            )
            for k in value_to_indices.get(target, ()):
                if k != i and k != j:
                    i_s, j_s, k_s = sorted((i, j, k))
                    return True, (i_s, j_s, k_s)
    return False, None


def generate_instance(
    length: int = 12,
    dimension: int = 3,
    mod: int = 10,
    target_has_3sum: Optional[bool] = None,
    rng: Optional[random.Random] = None,
) -> Instance3Sum:
    """Generate a single 3SUM instance with specified length and dimension.

    If ``target_has_3sum`` is True, guarantees a solution exists. If False,
    guarantees no solution exists via corruption/resampling. If None, chooses
    the target class with 50% probability.
    """
    if mod != 10:
        raise ValueError(
            f"Modulus other than 10 is not supported in Experiment 0, got mod={mod}"
        )
    if rng is None:
        rng = random.Random()

    if target_has_3sum is None:
        target_has_3sum = rng.random() < 0.5

    max_attempts = 1000

    if target_has_3sum:
        for _ in range(max_attempts):
            i, j, k = sorted(rng.sample(range(length), 3))
            tuples = [
                tuple(rng.randrange(mod) for _ in range(dimension))
                for _ in range(length)
            ]
            target_k = tuple(
                (-tuples[i][d] - tuples[j][d]) % mod
                for d in range(dimension)
            )
            tuples[k] = target_k

            has_sol, indices = check_3sum(tuples, mod=mod)
            if has_sol:
                return Instance3Sum(
                    tuples=tuples,
                    has_3sum=True,
                    matching_indices=indices,
                )
        raise RuntimeError(
            "Failed to generate a positive 3SUM instance within max attempts"
        )

    for _ in range(max_attempts):
        tuples = [
            tuple(rng.randrange(mod) for _ in range(dimension))
            for _ in range(length)
        ]
        has_sol, indices = check_3sum(tuples, mod=mod)
        if not has_sol:
            return Instance3Sum(
                tuples=tuples,
                has_3sum=False,
                matching_indices=None,
            )

        assert indices is not None
        c_idx = indices[2]
        tuples[c_idx] = tuple(
            (tuples[c_idx][d] + rng.randint(1, mod - 1)) % mod
            for d in range(dimension)
        )
        has_sol, indices = check_3sum(tuples, mod=mod)
        if not has_sol:
            return Instance3Sum(
                tuples=tuples,
                has_3sum=False,
                matching_indices=None,
            )

    raise RuntimeError(
        "Failed to generate a negative 3SUM instance within max attempts"
    )
