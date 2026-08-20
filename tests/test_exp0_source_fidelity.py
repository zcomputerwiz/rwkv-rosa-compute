"""Regression gates for Match-3 source distribution and dense-CoT fidelity."""

import random

import pytest

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, build_default_vocab, generate_packed_instances
from exp0.evaluate import compute_run_id
from exp0.sequences import format_a_parallel_cot
from exp0.task3sum import (
    LEGACY_GENERATOR,
    SOURCE_GENERATOR,
    Instance3Sum,
    check_3sum,
    generate_instance,
    matching_k_after_pair,
)

pytestmark = pytest.mark.exp0


def _oracle_source_instance(
    *,
    length: int,
    dimension: int,
    target: bool,
    seed: int,
    corruption_rate: float = 4 / 3,
) -> Instance3Sum:
    """Independent transcription of JacobPfau/fillerTokens Match3 generation."""
    rng = random.Random(seed)

    def rand_tuple():
        return tuple(rng.randrange(10) for _ in range(dimension))

    def planted():
        first = rand_tuple()
        second = rand_tuple()
        inverse = tuple((-first[d] - second[d]) % 10 for d in range(dimension))
        values = [first, second, inverse]
        values.extend(rand_tuple() for _ in range(length - 3))
        return values

    if target:
        values = planted()
        rng.shuffle(values)
        solved, indices = check_3sum(values)
        assert solved and indices is not None
        return Instance3Sum(values, True, indices)

    p = 1 / corruption_rate
    for _ in range(1000):
        values = [list(value) for value in planted()]
        corruptions = 1
        while corruptions < 3 and rng.random() >= p:
            corruptions += 1

        # Source: inputs[:corruptions, columns] = random_values. With a slice on
        # rows and an advanced index on columns, NumPy broadcasts each selected
        # column/value across all first `corruptions` rows.
        columns = [rng.randrange(dimension) for _ in range(corruptions)]
        replacements = [rng.randrange(10) for _ in range(corruptions)]
        for column, replacement in zip(columns, replacements):
            for row in range(corruptions):
                values[row][column] = replacement

        tuples = [tuple(value) for value in values]
        rng.shuffle(tuples)
        solved, _ = check_3sum(tuples)
        if not solved:
            return Instance3Sum(tuples, False, None)
    raise AssertionError("oracle failed to generate a negative")


@pytest.mark.parametrize("target", [False, True])
def test_source_generator_matches_independent_oracle(target):
    for seed in range(20):
        expected = _oracle_source_instance(
            length=6,
            dimension=3,
            target=target,
            seed=seed,
        )
        actual = generate_instance(
            length=6,
            dimension=3,
            target_has_3sum=target,
            rng=random.Random(seed),
            generator_mode=SOURCE_GENERATOR,
        )
        assert actual == expected


def test_legacy_generator_remains_explicitly_available():
    source = generate_instance(
        length=6,
        dimension=3,
        target_has_3sum=False,
        rng=random.Random(123),
        generator_mode=SOURCE_GENERATOR,
    )
    legacy = generate_instance(
        length=6,
        dimension=3,
        target_has_3sum=False,
        rng=random.Random(123),
        generator_mode=LEGACY_GENERATOR,
    )
    assert source.has_3sum is False
    assert legacy.has_3sum is False
    assert source.tuples != legacy.tuples


def test_source_match_policy_exposes_ordered_solution_once():
    instance = Instance3Sum(
        tuples=[(1, 0), (2, 0), (7, 0), (4, 4)],
        has_3sum=True,
        matching_indices=(0, 1, 2),
    )

    sum_ab, match_ab = matching_k_after_pair(instance.tuples, 0, 1)
    assert sum_ab == (3, 0)
    assert match_ab == 2
    assert matching_k_after_pair(instance.tuples, 0, 2)[1] is None
    assert matching_k_after_pair(instance.tuples, 1, 2)[1] is None

    tokens = format_a_parallel_cot(
        instance,
        vocab_reduction=False,
        rng=random.Random(7),
    ).split()
    separator = tokens.index(":")
    continuation = tokens[separator + 1 : -2]
    assert continuation.count("C") == 1


def test_reduced_tensor_hot_path_matches_formatter_exactly():
    source_rng = random.Random(321)
    instances = [
        generate_instance(
            length=6,
            dimension=3,
            rng=source_rng,
            generator_mode=SOURCE_GENERATOR,
        )
        for _ in range(12)
    ]
    vocab = build_default_vocab(length=6, dimension=3)
    dataset = Task3SumDataset(
        instances,
        format_type="parallel_cot",
        vocab=vocab,
        seed=91,
        vocab_reduction=True,
    )

    for idx, instance in enumerate(instances):
        expected = format_a_parallel_cot(
            instance,
            vocab_reduction=True,
            rng=random.Random(f"91_{idx}"),
        ).split()
        expected = expected[expected.index(":") :]
        actual = vocab.decode(dataset[idx]["targets"].tolist())
        assert actual == expected


def test_packed_generation_respects_generator_mode():
    source = generate_packed_instances(
        32,
        length=6,
        dimension=3,
        rng=random.Random(55),
        generator_mode=SOURCE_GENERATOR,
    )
    legacy = generate_packed_instances(
        32,
        length=6,
        dimension=3,
        rng=random.Random(55),
        generator_mode=LEGACY_GENERATOR,
    )
    assert source.tuples.shape == legacy.tuples.shape
    assert not source.tuples.equal(legacy.tuples)
    for packed in (source, legacy):
        for idx in range(len(packed)):
            solved, _ = check_3sum(
                [tuple(row) for row in packed.tuples[idx].tolist()]
            )
            assert solved == bool(packed.has_3sum[idx].item())


def test_generator_protocol_changes_run_identity():
    model = ModelConfig()
    train = TrainConfig()
    source = Task3SumConfig(generator_mode=SOURCE_GENERATOR)
    legacy = Task3SumConfig(generator_mode=LEGACY_GENERATOR)
    source_id = compute_run_id(model, train, source, 9999, 2000, [42])
    legacy_id = compute_run_id(model, train, legacy, 9999, 2000, [42])
    assert source_id != legacy_id
