"""3SUM task instance generation and solving logic."""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

SOURCE_GENERATOR = "source_corrupted"
LEGACY_GENERATOR = "uniform_conditioned"
GENERATOR_MODES = (SOURCE_GENERATOR, LEGACY_GENERATOR)
DEFAULT_CORRUPTION_RATE = 4 / 3


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
    """Return the first matching triple using exact legacy search ordering."""
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


def matching_k_after_pair(
    tuples: List[Tuple[int, ...]],
    i: int,
    j: int,
    mod: int = 10,
) -> tuple[Tuple[int, ...], Optional[int]]:
    """Return pair sum and first source-faithful matching ``k > j``."""
    dimension = len(tuples[0])
    sum_ij = tuple(
        (tuples[i][dim] + tuples[j][dim]) % mod
        for dim in range(dimension)
    )
    target = tuple((-value) % mod for value in sum_ij)
    for k in range(j + 1, len(tuples)):
        if tuples[k] == target:
            return sum_ij, k
    return sum_ij, None


def _validate_generation_args(
    length: int,
    dimension: int,
    mod: int,
    corruption_rate: float,
) -> None:
    if mod != 10:
        raise ValueError(
            f"Modulus other than 10 is not supported in Experiment 0, got mod={mod}"
        )
    if length < 3:
        raise ValueError("Match-3 generation requires length >= 3.")
    if dimension <= 0:
        raise ValueError("Match-3 generation requires dimension > 0.")
    if corruption_rate < 1.0:
        raise ValueError("corruption_rate must be >= 1.0.")


def _random_tuple(
    dimension: int,
    mod: int,
    rng: random.Random,
) -> Tuple[int, ...]:
    return tuple(rng.randrange(mod) for _ in range(dimension))


def _source_planted_tuples(
    length: int,
    dimension: int,
    mod: int,
    rng: random.Random,
) -> List[Tuple[int, ...]]:
    """Construct the source implementation's planted Match-3 base example."""
    first = _random_tuple(dimension, mod, rng)
    second = _random_tuple(dimension, mod, rng)
    inverse = tuple(
        (-first[dim] - second[dim]) % mod
        for dim in range(dimension)
    )
    tuples = [first, second, inverse]
    tuples.extend(
        _random_tuple(dimension, mod, rng)
        for _ in range(length - 3)
    )
    return tuples


def _capped_geometric(
    success_probability: float,
    cap: int,
    rng: random.Random,
) -> int:
    """Sample ``min(Geometric(p), cap)`` without a NumPy dependency."""
    value = 1
    while value < cap and rng.random() >= success_probability:
        value += 1
    return value


def _apply_source_corruption(
    tuples: List[Tuple[int, ...]],
    corruptions: int,
    dimension: int,
    mod: int,
    rng: random.Random,
) -> List[Tuple[int, ...]]:
    """Mirror the source NumPy advanced-index corruption semantics.

    The published code assigns ``inputs[:corruptions, columns] = values``.
    Because ``columns`` is an advanced index, every selected column is changed
    across every one of the first ``corruptions`` rows; the value vector is
    broadcast across rows. Repeated columns are resolved by later assignments.
    """
    mutable = [list(value) for value in tuples]
    columns = [rng.randrange(dimension) for _ in range(corruptions)]
    values = [rng.randrange(mod) for _ in range(corruptions)]
    for column, value in zip(columns, values):
        for row in range(corruptions):
            mutable[row][column] = value
    return [tuple(value) for value in mutable]


def generate_source_instance(
    length: int = 12,
    dimension: int = 3,
    mod: int = 10,
    target_has_3sum: Optional[bool] = None,
    rng: Optional[random.Random] = None,
    corruption_rate: float = DEFAULT_CORRUPTION_RATE,
) -> Instance3Sum:
    """Generate from the published positive-control construction.

    Positive examples plant a valid triple, append random tuples, and shuffle.
    Negative examples begin from the same planted construction, apply the
    source geometric corruption rule to the planted rows, shuffle, and
    reject/resample if any solution remains. The rejection preserves this
    harness's explicit ``target_has_3sum=False`` contract while matching the
    source construction and corruption distribution.
    """
    _validate_generation_args(length, dimension, mod, corruption_rate)
    if rng is None:
        rng = random.Random()
    if target_has_3sum is None:
        target_has_3sum = rng.random() < 0.5

    max_attempts = 1000
    if target_has_3sum:
        tuples = _source_planted_tuples(length, dimension, mod, rng)
        rng.shuffle(tuples)
        has_sol, indices = check_3sum(tuples, mod=mod)
        if not has_sol or indices is None:
            raise AssertionError("Planted positive Match-3 instance lost its solution.")
        return Instance3Sum(
            tuples=tuples,
            has_3sum=True,
            matching_indices=indices,
        )

    success_probability = 1.0 / corruption_rate
    for _ in range(max_attempts):
        planted = _source_planted_tuples(length, dimension, mod, rng)
        corruptions = _capped_geometric(success_probability, 3, rng)
        tuples = _apply_source_corruption(
            planted,
            corruptions,
            dimension,
            mod,
            rng,
        )
        rng.shuffle(tuples)

        has_sol, _ = check_3sum(tuples, mod=mod)
        if not has_sol:
            return Instance3Sum(
                tuples=tuples,
                has_3sum=False,
                matching_indices=None,
            )

    raise RuntimeError(
        "Failed to generate a source-style negative Match-3 instance within "
        "max attempts"
    )


def generate_uniform_conditioned_instance(
    length: int = 12,
    dimension: int = 3,
    mod: int = 10,
    target_has_3sum: Optional[bool] = None,
    rng: Optional[random.Random] = None,
) -> Instance3Sum:
    """Generate using the pre-fidelity Experiment-0 conditioned distribution."""
    _validate_generation_args(
        length,
        dimension,
        mod,
        DEFAULT_CORRUPTION_RATE,
    )
    if rng is None:
        rng = random.Random()

    if target_has_3sum is None:
        target_has_3sum = rng.random() < 0.5

    max_attempts = 1000

    if target_has_3sum:
        for _ in range(max_attempts):
            i, j, k = sorted(rng.sample(range(length), 3))
            tuples = [_random_tuple(dimension, mod, rng) for _ in range(length)]
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
        tuples = [_random_tuple(dimension, mod, rng) for _ in range(length)]
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
        has_sol, _ = check_3sum(tuples, mod=mod)
        if not has_sol:
            return Instance3Sum(
                tuples=tuples,
                has_3sum=False,
                matching_indices=None,
            )

    raise RuntimeError(
        "Failed to generate a negative 3SUM instance within max attempts"
    )


def generate_instance(
    length: int = 12,
    dimension: int = 3,
    mod: int = 10,
    target_has_3sum: Optional[bool] = None,
    rng: Optional[random.Random] = None,
    *,
    generator_mode: str = SOURCE_GENERATOR,
    corruption_rate: float = DEFAULT_CORRUPTION_RATE,
) -> Instance3Sum:
    """Generate one instance under an explicit, provenance-worthy distribution."""
    if generator_mode == SOURCE_GENERATOR:
        return generate_source_instance(
            length=length,
            dimension=dimension,
            mod=mod,
            target_has_3sum=target_has_3sum,
            rng=rng,
            corruption_rate=corruption_rate,
        )
    if generator_mode == LEGACY_GENERATOR:
        return generate_uniform_conditioned_instance(
            length=length,
            dimension=dimension,
            mod=mod,
            target_has_3sum=target_has_3sum,
            rng=rng,
        )
    raise ValueError(
        f"Unknown generator_mode={generator_mode!r}; expected one of {GENERATOR_MODES}."
    )
