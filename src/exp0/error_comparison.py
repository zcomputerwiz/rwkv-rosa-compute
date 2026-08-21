"""Cross-seed error-set comparison for Experiment 0 diagnostic artifacts."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp0.checkpoint_analysis import DIAGNOSTIC_ARTIFACT_VERSION

COMPARISON_ARTIFACT_VERSION = 1


def hypergeometric_upper_tail(
    population_size: int,
    successes: int,
    draws: int,
    observed: int,
) -> float:
    """Exact ``P(X >= observed)`` for a hypergeometric random variable."""
    for name, value in (
        ("population_size", population_size),
        ("successes", successes),
        ("draws", draws),
        ("observed", observed),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    if successes > population_size or draws > population_size:
        raise ValueError("successes and draws must not exceed population_size.")
    low = max(0, draws - (population_size - successes))
    high = min(successes, draws)
    if observed <= low:
        return 1.0
    if observed > high:
        return 0.0
    denominator = math.comb(population_size, draws)
    probability = math.fsum(
        (
            math.comb(successes, intersection)
            * math.comb(population_size - successes, draws - intersection)
        )
        / denominator
        for intersection in range(observed, high + 1)
    )
    return min(1.0, probability)


def load_diagnostic_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "exp0_checkpoint_diagnostics":
        raise ValueError(f"{source} is not an Experiment 0 diagnostic artifact.")
    if payload.get("artifact_version") != DIAGNOSTIC_ARTIFACT_VERSION:
        raise ValueError(
            f"{source} has unsupported artifact version "
            f"{payload.get('artifact_version')!r}."
        )
    payload["_source_path"] = str(source)
    return payload


def _seed(artifact: Mapping[str, Any]) -> int:
    return int(artifact["checkpoint"]["training_seed"])


def _records_by_index(section: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record in section["records"]:
        index = int(record["index"])
        if index in records:
            raise ValueError(f"Duplicate example index {index} in diagnostic artifact.")
        records[index] = record
    expected = set(range(int(section["population_size"])))
    if set(records) != expected:
        raise ValueError("Diagnostic records do not cover the complete population.")
    return records


def _normalized_training_protocol(artifact: Mapping[str, Any]) -> dict[str, Any]:
    protocol = dict(artifact["provenance"]["training_protocol"])
    protocol.pop("seed", None)
    return protocol


def _normalized_model_config(artifact: Mapping[str, Any]) -> dict[str, Any]:
    model = dict(artifact["provenance"]["model_config"])
    model.pop("device", None)
    return model


def _evaluation_semantics(artifact: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = artifact["provenance"]["evaluation"]
    return {key: evaluation.get(key) for key in ("device", "precision", "batch_size")}


def _validate_compatibility(artifacts: Sequence[Mapping[str, Any]]) -> None:
    if len(artifacts) < 2:
        raise ValueError("At least two diagnostic artifacts are required.")
    seeds = [_seed(artifact) for artifact in artifacts]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Training seeds must be unique, found {seeds}.")

    first = artifacts[0]
    expected_id = first["canonical_validation"]["canonical_validation_id"]
    expected_generation = first["canonical_validation"]["generation_config"]
    expected_task = first["provenance"]["task_config"]
    expected_model = _normalized_model_config(first)
    expected_training = _normalized_training_protocol(first)
    expected_evaluation = _evaluation_semantics(first)
    for artifact in artifacts[1:]:
        if artifact["canonical_validation"]["canonical_validation_id"] != expected_id:
            raise ValueError("canonical_validation_id mismatch across artifacts.")
        if artifact["canonical_validation"]["generation_config"] != expected_generation:
            raise ValueError("Canonical evaluation configuration mismatch.")
        if artifact["provenance"]["task_config"] != expected_task:
            raise ValueError("Task configuration mismatch across artifacts.")
        if _normalized_model_config(artifact) != expected_model:
            raise ValueError("Model configuration mismatch across artifacts.")
        if _normalized_training_protocol(artifact) != expected_training:
            raise ValueError("Training protocol mismatch across artifacts.")
        if _evaluation_semantics(artifact) != expected_evaluation:
            raise ValueError("Evaluation execution configuration mismatch.")

    challenge_presence = [
        "diagnostic_challenge_validation" in artifact for artifact in artifacts
    ]
    if any(challenge_presence) and not all(challenge_presence):
        raise ValueError("Challenge diagnostics are missing from some artifacts.")
    if all(challenge_presence):
        expected_challenge_id = first["diagnostic_challenge_validation"]["challenge_id"]
        expected_content = first["diagnostic_challenge_validation"][
            "challenge_content_sha256"
        ]
        for artifact in artifacts[1:]:
            challenge = artifact["diagnostic_challenge_validation"]
            if challenge["challenge_id"] != expected_challenge_id:
                raise ValueError("challenge_id mismatch across artifacts.")
            if challenge["challenge_content_sha256"] != expected_content:
                raise ValueError("Challenge contents differ despite matching specs.")


def _error_set(
    records: Mapping[int, Mapping[str, Any]],
    population: set[int],
) -> set[int]:
    return {index for index in population if not records[index]["correct"]}


def _pairwise_overlap(
    records_by_seed: Mapping[int, Mapping[int, Mapping[str, Any]]],
    population: set[int],
) -> list[dict[str, Any]]:
    rows = []
    for seed_a, seed_b in itertools.combinations(sorted(records_by_seed), 2):
        errors_a = _error_set(records_by_seed[seed_a], population)
        errors_b = _error_set(records_by_seed[seed_b], population)
        intersection = errors_a & errors_b
        union = errors_a | errors_b
        expected = (
            len(errors_a) * len(errors_b) / len(population) if population else 0.0
        )
        rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "population_size": len(population),
                "errors_a": len(errors_a),
                "errors_b": len(errors_b),
                "observed_intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else 1.0,
                "expected_independent_intersection": expected,
                "observed_expected_enrichment": (
                    len(intersection) / expected if expected else None
                ),
                "hypergeometric_upper_tail_p": hypergeometric_upper_tail(
                    len(population),
                    len(errors_a),
                    len(errors_b),
                    len(intersection),
                )
                if population
                else 1.0,
            }
        )
    return rows


def _per_seed_summary(
    records_by_seed: Mapping[int, Mapping[int, Mapping[str, Any]]],
    population: set[int],
) -> list[dict[str, Any]]:
    rows = []
    for seed in sorted(records_by_seed):
        errors = _error_set(records_by_seed[seed], population)
        rows.append(
            {
                "seed": seed,
                "population_size": len(population),
                "error_count": len(errors),
                "error_rate": len(errors) / len(population) if population else None,
                "error_indices": sorted(errors),
            }
        )
    return rows


def _metadata_without_prediction(record: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "correct",
        "predicted_label",
        "true_logit",
        "false_logit",
        "prediction_margin",
    }
    return {key: value for key, value in record.items() if key not in excluded}


def _persistence(
    records_by_seed: Mapping[int, Mapping[int, Mapping[str, Any]]],
    population: set[int],
) -> tuple[dict[str, int], list[dict[str, Any]], dict[int, int]]:
    seeds = sorted(records_by_seed)
    frequency: dict[int, int] = {}
    for index in sorted(population):
        frequency[index] = sum(
            not records_by_seed[seed][index]["correct"] for seed in seeds
        )
    histogram = {
        str(count): sum(value == count for value in frequency.values())
        for count in range(len(seeds) + 1)
    }
    persistent = []
    for index, count in sorted(frequency.items()):
        if count < 2:
            continue
        baseline = records_by_seed[seeds[0]][index]
        persistent.append(
            {
                **_metadata_without_prediction(baseline),
                "missed_by_seed_count": count,
                "per_seed": {
                    str(seed): {
                        "correct": records_by_seed[seed][index]["correct"],
                        "predicted_label": records_by_seed[seed][index][
                            "predicted_label"
                        ],
                        "prediction_margin": records_by_seed[seed][index][
                            "prediction_margin"
                        ],
                    }
                    for seed in seeds
                },
            }
        )
    return histogram, persistent, frequency


def _stratum_population(
    records_by_seed: Mapping[int, Mapping[int, Mapping[str, Any]]],
    field: str,
    value: str,
) -> set[int]:
    seeds = sorted(records_by_seed)
    populations = [
        {
            index
            for index, record in records_by_seed[seed].items()
            if record.get(field) == value
        }
        for seed in seeds
    ]
    if any(population != populations[0] for population in populations[1:]):
        raise ValueError(f"Population metadata mismatch for {field}={value}.")
    return populations[0]


def _stratum_analysis(
    records_by_seed: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    first_records = records_by_seed[sorted(records_by_seed)[0]]
    result: dict[str, Any] = {}
    for field, group_name in (
        ("primary_stratum", "construction_strata"),
        ("corruption_stratum", "corruption_strata"),
    ):
        values = sorted(
            {
                record.get(field)
                for record in first_records.values()
                if record.get(field) is not None
            }
        )
        result[group_name] = {}
        for value in values:
            population = _stratum_population(records_by_seed, field, value)
            result[group_name][value] = {
                "population_size": len(population),
                "descriptive_only": len(population) < 30,
                "per_seed": _per_seed_summary(records_by_seed, population),
                "pairwise": _pairwise_overlap(records_by_seed, population),
            }
    return result


def _distribution(values: Sequence[int | float | None]) -> dict[str, Any]:
    present = [value for value in values if value is not None]
    histogram = Counter("null" if value is None else str(value) for value in values)
    return {
        "count": len(values),
        "null_count": len(values) - len(present),
        "histogram": dict(sorted(histogram.items(), key=lambda item: item[0])),
        "mean": statistics.fmean(present) if present else None,
        "median": statistics.median(present) if present else None,
        "minimum": min(present) if present else None,
        "maximum": max(present) if present else None,
    }


def _feature_distributions(
    records: Mapping[int, Mapping[str, Any]],
    population: set[int],
) -> dict[str, Any]:
    scalar_features = (
        "num_valid_triples",
        "first_valid_witness_position",
        "num_two_of_three_near_misses",
        "max_matched_coordinate_count_among_non_solutions",
    )
    result = {
        feature: _distribution([records[index].get(feature) for index in population])
        for feature in scalar_features
    }
    coordinate_width = max(
        (
            len(records[index].get("candidate_coordinate_match_counts", []))
            for index in population
        ),
        default=0,
    )
    for matched in range(coordinate_width):
        result[f"candidate_triples_matching_{matched}_coordinates"] = _distribution(
            [
                records[index]["candidate_coordinate_match_counts"][matched]
                for index in population
            ]
        )
    return result


def _structural_analysis(
    records_by_seed: Mapping[int, Mapping[int, Mapping[str, Any]]],
    population: set[int],
    miss_frequency: Mapping[int, int],
) -> dict[str, Any]:
    baseline = records_by_seed[sorted(records_by_seed)[0]]
    groups = {
        "correct_in_all_seeds": {
            index for index in population if miss_frequency[index] == 0
        },
        "missed_by_exactly_one_seed": {
            index for index in population if miss_frequency[index] == 1
        },
        "missed_by_at_least_two_seeds": {
            index for index in population if miss_frequency[index] >= 2
        },
    }

    def summarize(group_population: set[int]) -> dict[str, Any]:
        return {
            "population_size": len(group_population),
            "features": _feature_distributions(baseline, group_population),
        }

    overall = {name: summarize(indices) for name, indices in groups.items()}
    corrupted_negative = {
        index
        for index in population
        if baseline[index].get("primary_stratum") == "corrupted_arm_negative"
    }
    corrupted = {
        name: summarize(indices & corrupted_negative)
        for name, indices in groups.items()
    }
    return {
        "overall": overall,
        "corrupted_arm_negative": corrupted,
        "note": (
            "Features are deterministic functions of the mathematical instance; "
            "group membership is post-hoc model behavior. Histograms are the "
            "primary output and means are descriptive summaries only."
        ),
    }


def _population_analysis(
    artifacts: Sequence[Mapping[str, Any]],
    section_name: str,
) -> dict[str, Any]:
    records_by_seed = {
        _seed(artifact): _records_by_index(artifact[section_name])
        for artifact in artifacts
    }
    first_seed = sorted(records_by_seed)[0]
    population = set(records_by_seed[first_seed])
    for seed, records in records_by_seed.items():
        if set(records) != population:
            raise ValueError(f"Population indices differ for seed {seed}.")
        for index in population:
            if (
                records[index]["example_id"]
                != records_by_seed[first_seed][index]["example_id"]
            ):
                raise ValueError(
                    f"Example identity mismatch at index {index} for seed {seed}."
                )
    histogram, persistent, frequency = _persistence(records_by_seed, population)
    return {
        "per_seed": _per_seed_summary(records_by_seed, population),
        "pairwise": _pairwise_overlap(records_by_seed, population),
        "miss_frequency": histogram,
        "persistent_errors": persistent,
        "strata": _stratum_analysis(records_by_seed),
        "structural_analysis": _structural_analysis(
            records_by_seed, population, frequency
        ),
        "_records_by_seed": records_by_seed,
        "_population": population,
    }


def _reference_analysis(
    records_by_seed: Mapping[int, Mapping[int, Mapping[str, Any]]],
    population: set[int],
    reference_indices: set[int],
) -> dict[str, Any]:
    unknown = sorted(reference_indices - population)
    if unknown:
        raise ValueError(
            f"Reference error indices are outside the population: {unknown}"
        )
    seeds = sorted(records_by_seed)
    table = []
    for index in sorted(reference_indices):
        missed = {
            str(seed): not records_by_seed[seed][index]["correct"] for seed in seeds
        }
        table.append(
            {
                "index": index,
                "per_seed_missed": missed,
                "recurrence": sum(missed.values()),
            }
        )
    per_seed = []
    for seed in seeds:
        errors = _error_set(records_by_seed[seed], population)
        observed = len(errors & reference_indices)
        expected = len(errors) * len(reference_indices) / len(population)
        per_seed.append(
            {
                "seed": seed,
                "observed_recurrence": observed,
                "expected_recurrence": expected,
                "observed_expected_enrichment": (
                    observed / expected if expected else None
                ),
                "hypergeometric_upper_tail_p": hypergeometric_upper_tail(
                    len(population),
                    len(errors),
                    len(reference_indices),
                    observed,
                ),
                "recurrent_indices": sorted(errors & reference_indices),
            }
        )
    return {
        "reference_indices": sorted(reference_indices),
        "per_seed": per_seed,
        "recurrence_table": table,
    }


def compare_diagnostic_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    reference_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compare two or more compatible checkpoint diagnostic artifacts."""
    ordered = sorted(artifacts, key=_seed)
    _validate_compatibility(ordered)
    canonical = _population_analysis(ordered, "canonical_validation")
    canonical_records = canonical.pop("_records_by_seed")
    canonical_population = canonical.pop("_population")

    result: dict[str, Any] = {
        "artifact_type": "exp0_cross_seed_error_comparison",
        "artifact_version": COMPARISON_ARTIFACT_VERSION,
        "seeds": [_seed(artifact) for artifact in ordered],
        "sources": [artifact.get("_source_path") for artifact in ordered],
        "compatibility": {
            "canonical_validation_id": ordered[0]["canonical_validation"][
                "canonical_validation_id"
            ],
            "task_config": ordered[0]["provenance"]["task_config"],
            "model_config": _normalized_model_config(ordered[0]),
            "training_protocol": _normalized_training_protocol(ordered[0]),
            "evaluation_execution": _evaluation_semantics(ordered[0]),
        },
        "canonical_validation": canonical,
    }
    if reference_indices is not None:
        result["reference_error_set"] = _reference_analysis(
            canonical_records,
            canonical_population,
            {int(index) for index in reference_indices},
        )

    if "diagnostic_challenge_validation" in ordered[0]:
        challenge = _population_analysis(ordered, "diagnostic_challenge_validation")
        challenge.pop("_records_by_seed")
        challenge.pop("_population")
        challenge["challenge_id"] = ordered[0]["diagnostic_challenge_validation"][
            "challenge_id"
        ]
        challenge["challenge_content_sha256"] = ordered[0][
            "diagnostic_challenge_validation"
        ]["challenge_content_sha256"]
        result["diagnostic_challenge_validation"] = challenge
    return result


def write_comparison_artifact(
    comparison: Mapping[str, Any],
    path: str | Path,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target

