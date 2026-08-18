"""3SUM task instance generation and solving logic."""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Instance3Sum:
    """A 3SUM instance consisting of n d-dimensional tuples in Z_10^d."""

    tuples: List[Tuple[int, ...]]  # List of length n, each element is a d-tuple of ints in 0..9
    has_3sum: bool
    matching_indices: Optional[Tuple[int, int, int]] = None  # (i, j, k) with i < j < k if found


def check_3sum(tuples: List[Tuple[int, ...]], mod: int = 10) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
    """Check if there exist distinct i, j, k such that tuples[i] + tuples[j] + tuples[k] == 0 (mod mod)."""
    n = len(tuples)
    d = len(tuples[0])

    # Map target complement tuple -> index
    # We want (x_i + x_j + x_k) % mod == 0 => x_k = (-x_i - x_j) % mod
    for i in range(n):
        for j in range(i + 1, n):
            target = tuple((-tuples[i][dim] - tuples[j][dim]) % mod for dim in range(d))
            for k in range(n):
                if k != i and k != j:
                    if tuples[k] == target:
                        # Return sorted indices
                        i_s, j_s, k_s = sorted([i, j, k])
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

    If target_has_3sum is True, guarantees a solution exists.
    If target_has_3sum is False, guarantees no solution exists via corruption/resampling.
    If None, picks randomly with 50% target.
    """
    if rng is None:
        rng = random.Random()

    if target_has_3sum is None:
        target_has_3sum = rng.random() < 0.5

    max_attempts = 1000

    if target_has_3sum:
        for _ in range(max_attempts):
            # Pick 3 random distinct indices to hold a solution
            i, j, k = sorted(rng.sample(range(length), 3))

            # Draw random tuples for all positions except k
            tuples = [tuple(rng.randrange(mod) for _ in range(dimension)) for _ in range(length)]

            # Force tuples[k] = (-tuples[i] - tuples[j]) % mod
            target_k = tuple((-tuples[i][d] - tuples[j][d]) % mod for d in range(dimension))
            tuples[k] = target_k

            has_sol, indices = check_3sum(tuples, mod=mod)
            if has_sol:
                return Instance3Sum(tuples=tuples, has_3sum=True, matching_indices=indices)
        raise RuntimeError("Failed to generate a positive 3SUM instance within max attempts")
    else:
        for _ in range(max_attempts):
            tuples = [tuple(rng.randrange(mod) for _ in range(dimension)) for _ in range(length)]
            has_sol, indices = check_3sum(tuples, mod=mod)
            if not has_sol:
                return Instance3Sum(tuples=tuples, has_3sum=False, matching_indices=None)

            # If accidentally generated a solution, corrupt one element involved in the solution
            assert indices is not None
            c_idx = indices[2]
            tuples[c_idx] = tuple((tuples[c_idx][d] + rng.randint(1, mod - 1)) % mod for d in range(dimension))
            has_sol, indices = check_3sum(tuples, mod=mod)
            if not has_sol:
                return Instance3Sum(tuples=tuples, has_3sum=False, matching_indices=None)

        raise RuntimeError("Failed to generate a negative 3SUM instance within max attempts")
