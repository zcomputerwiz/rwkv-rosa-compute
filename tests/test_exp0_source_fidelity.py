"""Regression gates for Match-3 source distribution and dense-CoT fidelity."""

import random

import pytest
import torch

import exp0.generation as generation
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
    construction_positive: bool,
    seed: int,
    corruption_rate: float = 4 / 3,
) -> Instance3Sum:
    """Independent transcription of JacobPfau/fillerTokens Match3 construction."""
    rng = random.Random(seed)

    def rand_tuple():
        return tuple(rng.randrange(10) for _ in range(dimension))

    first = rand_tuple()
    second = rand_tuple()
    inverse = tuple((-first[d] - second[d]) % 10 for d in range(dimension))
    core = [first, second, inverse]

    if construction_positive:
        values = core + [rand_tuple() for _ in range(length - 3)]
        rng.shuffle(values)
        solved, indices = check_3sum(values)
        assert solved and indices is not None
        return Instance3Sum(
            values, True, indices, construction_arm=True, corruption_count=None
        )

    corruptions = 1
    p = 1 / corruption_rate
    while corruptions < 3 and rng.random() >= p:
        corruptions += 1

    values = [list(value) for value in core]
    columns = [rng.randrange(dimension) for _ in range(corruptions)]
    replacements = [rng.randrange(10) for _ in range(corruptions)]
    for column, replacement in zip(columns, replacements):
        for row in range(corruptions):
            values[row][column] = replacement

    tuples = [tuple(value) for value in values]
    tuples.extend(rand_tuple() for _ in range(length - 3))
    rng.shuffle(tuples)
    solved, indices = check_3sum(tuples)
    # The oracle knows the arm and corruption count independently, so the
    # comparison also verifies the recorded generation provenance.
    return Instance3Sum(
        tuples, solved, indices, construction_arm=False, corruption_count=corruptions
    )


@pytest.mark.parametrize("construction_positive", [False, True])
def test_source_generator_matches_independent_oracle(construction_positive):
    for seed in range(20):
        expected = _oracle_source_instance(
            length=6,
            dimension=3,
            construction_positive=construction_positive,
            seed=seed,
        )
        actual = generate_instance(
            length=6,
            dimension=3,
            target_has_3sum=construction_positive,
            rng=random.Random(seed),
            generator_mode=SOURCE_GENERATOR,
        )
        assert actual == expected


def test_source_corrupted_arm_can_still_have_positive_label():
    survivors = [
        generate_instance(
            length=6,
            dimension=3,
            target_has_3sum=False,
            rng=random.Random(seed),
            generator_mode=SOURCE_GENERATOR,
        ).has_3sum
        for seed in range(100)
    ]
    assert any(survivors)
    assert not all(survivors)


def test_protocol_generation_samples_construction_vector_before_tuple_contents(
    monkeypatch,
):
    seed = 4321
    count = 40
    true_rate = 0.3
    master = random.Random(seed)
    construction_rng = random.Random(master.getrandbits(128))
    _tuple_seed = master.getrandbits(128)
    expected_arms = [construction_rng.random() < true_rate for _ in range(count)]
    seen_arms = []

    def fake_generate_instance(
        length,
        dimension,
        mod,
        target_has_3sum,
        rng,
        *,
        generator_mode,
        corruption_rate,
    ):
        seen_arms.append(target_has_3sum)
        tuples = [(0,) * dimension for _ in range(length)]
        return Instance3Sum(tuples, bool(target_has_3sum), None)

    monkeypatch.setattr(generation, "generate_instance", fake_generate_instance)
    generation.generate_protocol_packed_instances(
        count,
        length=6,
        dimension=3,
        true_rate=true_rate,
        rng=random.Random(seed),
        generator_mode=SOURCE_GENERATOR,
    )

    assert seen_arms == expected_arms


def test_protocol_generation_has_dataset_size_stable_common_prefix():
    small = generation.generate_protocol_packed_instances(
        8,
        length=6,
        dimension=3,
        rng=random.Random(90210),
        generator_mode=SOURCE_GENERATOR,
    )
    large = generation.generate_protocol_packed_instances(
        12,
        length=6,
        dimension=3,
        rng=random.Random(90210),
        generator_mode=SOURCE_GENERATOR,
    )

    assert torch.equal(small.tuples, large.tuples[:8])
    assert torch.equal(small.has_3sum, large.has_3sum[:8])
    assert torch.equal(small.matching_indices, large.matching_indices[:8])


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
    source_solved, _ = check_3sum(source.tuples)
    assert source_solved == source.has_3sum
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


def test_reduced_tensor_hot_path_matches_formatter_exactly_plus_eos():
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
        expected.append(vocab.pad_token)
        actual = vocab.decode(dataset[idx]["targets"].tolist())
        assert actual == expected
        assert dataset[idx]["targets"][-1].item() == 0


def test_eos_target_protocol_is_required():
    with pytest.raises(ValueError, match="supervised EOS target"):
        Task3SumConfig(include_eos_target=False)
    with pytest.raises(ValueError, match="supervised EOS target"):
        Task3SumDataset([], include_eos_target=False)


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


def test_true_rate_changes_run_identity():
    model = ModelConfig()
    train = TrainConfig()
    balanced = Task3SumConfig(true_rate=0.5)
    shifted = Task3SumConfig(true_rate=0.4)
    assert compute_run_id(model, train, balanced, 9999, 2000, [42]) != compute_run_id(
        model, train, shifted, 9999, 2000, [42]
    )


def test_output_head_width_changes_run_identity():
    task = Task3SumConfig()
    train = TrainConfig()
    source_head = ModelConfig(output_vocab_size=32000)
    compact_head = ModelConfig(output_vocab_size=2048)
    assert compute_run_id(source_head, train, task, 9999, 2000, [42]) != compute_run_id(
        compact_head, train, task, 9999, 2000, [42]
    )
