"""CPU tests for Experiment 0 cross-seed hard-instance analysis."""

from copy import deepcopy

import pytest

from exp0.error_comparison import (
    compare_diagnostic_artifacts,
    hypergeometric_upper_tail,
)


def _artifact(seed, errors, *, canonical_id="canonical-same"):
    records = []
    strata = [
        "positive_arm_positive",
        "corrupted_arm_negative",
        "corrupted_arm_surviving_positive",
        "corrupted_arm_negative",
        "positive_arm_positive",
    ]
    for index in range(5):
        primary = strata[index]
        records.append(
            {
                "index": index,
                "example_id": f"same:{index}",
                "tuples": [[index, 0, 0]] * 3,
                "true_label": index % 2 == 0,
                "planted_matching_indices": None,
                "construction_arm": (
                    "corrupted" if primary.startswith("corrupted") else "positive"
                ),
                "corruption_count": 1 if primary.startswith("corrupted") else None,
                "correct": index not in errors,
                "predicted_label": index in errors,
                "true_logit": float(index),
                "false_logit": float(index + (index in errors)),
                "prediction_margin": -1.0 if index in errors else 0.0,
                "num_valid_triples": index % 2,
                "first_witness": None,
                "multiple_witnesses": False,
                "primary_stratum": primary,
                "corruption_stratum": (
                    "corrupted_negative_c1"
                    if primary == "corrupted_arm_negative"
                    else "corrupted_positive_c1"
                    if primary == "corrupted_arm_surviving_positive"
                    else None
                ),
                "total_candidate_triples": 1,
                "candidate_coordinate_match_counts": [0, 1, 0, 0],
                "num_two_of_three_near_misses": index,
                "max_matched_coordinate_count_among_non_solutions": 1,
                "first_valid_witness_position": None,
            }
        )
    return {
        "artifact_type": "exp0_checkpoint_diagnostics",
        "artifact_version": 1,
        "checkpoint": {"training_seed": seed},
        "provenance": {
            "model_config": {"architecture": "llama", "hidden_size": 12},
            "task_config": {"length": 3, "dimension": 3},
            "training_protocol": {"seed": seed, "precision": "fp32"},
            "evaluation": {"device": "cpu", "precision": "fp32", "batch_size": 2},
        },
        "canonical_validation": {
            "canonical_validation_id": canonical_id,
            "generation_config": {"eval_seed": 9999, "val_samples": 5},
            "population_size": 5,
            "records": records,
        },
    }


def test_hypergeometric_upper_tail_exact_small_case():
    assert hypergeometric_upper_tail(5, 2, 2, 1) == pytest.approx(0.7)
    assert hypergeometric_upper_tail(5, 2, 2, 0) == 1.0
    assert hypergeometric_upper_tail(5, 2, 2, 3) == 0.0


def test_pairwise_persistence_reference_and_deterministic_ordering():
    artifacts = [
        _artifact(45, {1, 3}),
        _artifact(43, {0, 1}),
        _artifact(44, {1, 2}),
    ]
    result = compare_diagnostic_artifacts(artifacts, reference_indices=[1, 4])
    canonical = result["canonical_validation"]
    assert result["seeds"] == [43, 44, 45]
    assert canonical["miss_frequency"] == {"0": 1, "1": 3, "2": 0, "3": 1}
    assert [row["index"] for row in canonical["persistent_errors"]] == [1]
    assert canonical["persistent_errors"][0]["missed_by_seed_count"] == 3

    pair = canonical["pairwise"][0]
    assert (pair["seed_a"], pair["seed_b"]) == (43, 44)
    assert pair["observed_intersection"] == 1
    assert pair["expected_independent_intersection"] == pytest.approx(0.8)
    assert pair["observed_expected_enrichment"] == pytest.approx(1.25)
    assert pair["jaccard"] == pytest.approx(1 / 3)
    assert pair["hypergeometric_upper_tail_p"] == pytest.approx(0.7)

    reference = result["reference_error_set"]
    assert [row["index"] for row in reference["recurrence_table"]] == [1, 4]
    assert reference["recurrence_table"][0]["recurrence"] == 3
    assert reference["recurrence_table"][1]["recurrence"] == 0
    assert (
        canonical["strata"]["construction_strata"]["corrupted_arm_negative"][
            "population_size"
        ]
        == 2
    )
    assert (
        canonical["structural_analysis"]["corrupted_arm_negative"][
            "missed_by_at_least_two_seeds"
        ]["population_size"]
        == 1
    )


def test_mismatched_validation_id_is_rejected():
    with pytest.raises(ValueError, match="canonical_validation_id mismatch"):
        compare_diagnostic_artifacts(
            [_artifact(43, {0}), _artifact(44, {1}, canonical_id="different")]
        )


def test_mismatched_challenge_id_is_rejected():
    first = _artifact(43, {0})
    second = _artifact(44, {1})
    for artifact, challenge_id in ((first, "a"), (second, "b")):
        challenge = deepcopy(artifact["canonical_validation"])
        challenge.pop("canonical_validation_id")
        challenge["challenge_id"] = challenge_id
        challenge["challenge_content_sha256"] = "content"
        artifact["diagnostic_challenge_validation"] = challenge
    with pytest.raises(ValueError, match="challenge_id mismatch"):
        compare_diagnostic_artifacts([first, second])

