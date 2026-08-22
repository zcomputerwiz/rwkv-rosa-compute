"""Structural challenge set generation partitioned by 2-of-3 near-match count."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

from exp0.checkpoint_analysis import structural_features
from exp0.task3sum import (
    DEFAULT_CORRUPTION_RATE,
    SOURCE_GENERATOR,
    generate_instance,
)

STRUCTURAL_CHALLENGE_VERSION = 1

STRATUM_POSITIVE_ARM_POSITIVE = "positive_arm_positive"
STRATUM_CORRUPTED_SURVIVING_POSITIVE = "corrupted_arm_surviving_positive"
STRATUM_CORRUPTED_NEGATIVE_NEAR_0 = "corrupted_negative_near_0"
STRATUM_CORRUPTED_NEGATIVE_NEAR_1 = "corrupted_negative_near_1"
STRATUM_CORRUPTED_NEGATIVE_NEAR_2 = "corrupted_negative_near_2"
STRATUM_CORRUPTED_NEGATIVE_NEAR_3PLUS = "corrupted_negative_near_3plus"

STRUCTURAL_STRATA: Tuple[str, ...] = (
    STRATUM_POSITIVE_ARM_POSITIVE,
    STRATUM_CORRUPTED_SURVIVING_POSITIVE,
    STRATUM_CORRUPTED_NEGATIVE_NEAR_0,
    STRATUM_CORRUPTED_NEGATIVE_NEAR_1,
    STRATUM_CORRUPTED_NEGATIVE_NEAR_2,
    STRATUM_CORRUPTED_NEGATIVE_NEAR_3PLUS,
)

DEFAULT_MAX_ATTEMPTS_PER_STRATUM = 500_000


@dataclass(frozen=True)
class StructuralChallengeSpec:
    """Specification for the 6-stratum structural challenge set."""

    seed: int = 20260821
    per_stratum: int = 1000
    length: int = 6
    dimension: int = 3
    mod: int = 10
    generator_mode: str = SOURCE_GENERATOR
    corruption_rate: float = DEFAULT_CORRUPTION_RATE
    max_attempts_per_stratum: int = DEFAULT_MAX_ATTEMPTS_PER_STRATUM
    strata: Tuple[str, ...] = STRUCTURAL_STRATA

    def __post_init__(self):
        if self.per_stratum <= 0:
            raise ValueError("per_stratum must be positive")
        if self.max_attempts_per_stratum <= 0:
            raise ValueError("max_attempts_per_stratum must be positive")
        unknown = set(self.strata) - set(STRUCTURAL_STRATA)
        if unknown:
            raise ValueError(f"unsupported challenge strata: {sorted(unknown)}")
        if not self.strata:
            raise ValueError("at least one stratum must be requested")

    def canonical_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["strata"] = list(self.strata)
        payload["structural_challenge_version"] = STRUCTURAL_CHALLENGE_VERSION
        payload.pop("max_attempts_per_stratum", None)
        return payload

    @property
    def challenge_id(self) -> str:
        blob = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def classify_structural_instance(instance, mod: int = 10) -> Tuple[str, Dict[str, Any]]:
    """Determine the structural challenge stratum and compute features for an instance."""
    tuples = [list(row) for row in instance.tuples]
    features = structural_features(tuples, mod=mod)
    has_3sum = bool(instance.has_3sum)
    is_positive_arm = bool(instance.construction_arm)

    if is_positive_arm:
        if has_3sum:
            return STRATUM_POSITIVE_ARM_POSITIVE, features
        else:
            raise ValueError("Positive arm generated an instance with no 3SUM solution.")

    # Corrupted arm
    if has_3sum:
        return STRATUM_CORRUPTED_SURVIVING_POSITIVE, features

    near_count = features["num_two_of_three_near_misses"] or 0
    if near_count == 0:
        stratum = STRATUM_CORRUPTED_NEGATIVE_NEAR_0
    elif near_count == 1:
        stratum = STRATUM_CORRUPTED_NEGATIVE_NEAR_1
    elif near_count == 2:
        stratum = STRATUM_CORRUPTED_NEGATIVE_NEAR_2
    else:
        stratum = STRATUM_CORRUPTED_NEGATIVE_NEAR_3PLUS

    return stratum, features


def generate_structural_challenge_set(spec: StructuralChallengeSpec) -> Dict[str, Any]:
    """Generate the stratum-balanced structural challenge set by rejection sampling."""
    wanted = {name: spec.per_stratum for name in spec.strata}
    accepted: Dict[str, List[Any]] = {name: [] for name in spec.strata}
    accepted_features: Dict[str, List[Dict[str, Any]]] = {name: [] for name in spec.strata}
    attempts: Dict[str, int] = {name: 0 for name in spec.strata}

    root = random.Random(spec.seed)
    arm_streams = {
        "positive": random.Random(root.getrandbits(128)),
        "corrupted": random.Random(root.getrandbits(128)),
    }

    def needed(names: Sequence[str]) -> List[str]:
        return [n for n in names if n in wanted and len(accepted[n]) < wanted[n]]

    # 1. Positive arm (feeds positive_arm_positive)
    if STRATUM_POSITIVE_ARM_POSITIVE in wanted:
        while len(accepted[STRATUM_POSITIVE_ARM_POSITIVE]) < wanted[STRATUM_POSITIVE_ARM_POSITIVE]:
            if attempts[STRATUM_POSITIVE_ARM_POSITIVE] >= spec.max_attempts_per_stratum:
                raise RuntimeError(
                    f"Exhausted attempt budget for {STRATUM_POSITIVE_ARM_POSITIVE}"
                )
            attempts[STRATUM_POSITIVE_ARM_POSITIVE] += 1
            instance = generate_instance(
                length=spec.length,
                dimension=spec.dimension,
                mod=spec.mod,
                target_has_3sum=True,
                rng=arm_streams["positive"],
                generator_mode=spec.generator_mode,
                corruption_rate=spec.corruption_rate,
            )
            stratum, feats = classify_structural_instance(instance, mod=spec.mod)
            if stratum == STRATUM_POSITIVE_ARM_POSITIVE:
                accepted[stratum].append(instance)
                accepted_features[stratum].append(feats)

    # 2. Corrupted arm (feeds surviving positives and near-match negative buckets)
    corrupted_strata = [s for s in spec.strata if s != STRATUM_POSITIVE_ARM_POSITIVE]
    while needed(corrupted_strata):
        for name in needed(corrupted_strata):
            if attempts[name] >= spec.max_attempts_per_stratum:
                raise RuntimeError(
                    f"Exhausted attempt budget for stratum {name}: "
                    f"filled {len(accepted[name])}/{wanted[name]} after {attempts[name]} attempts"
                )
            attempts[name] += 1

        instance = generate_instance(
            length=spec.length,
            dimension=spec.dimension,
            mod=spec.mod,
            target_has_3sum=False,
            rng=arm_streams["corrupted"],
            generator_mode=spec.generator_mode,
            corruption_rate=spec.corruption_rate,
        )
        stratum, feats = classify_structural_instance(instance, mod=spec.mod)
        if stratum in accepted and len(accepted[stratum]) < wanted[stratum]:
            accepted[stratum].append(instance)
            accepted_features[stratum].append(feats)

    # Assemble structured records in fixed stratum order
    ordered_records = []
    for stratum_name in spec.strata:
        instances = accepted[stratum_name]
        feats_list = accepted_features[stratum_name]
        for instance, feats in zip(instances, feats_list):
            ordered_records.append({
                "stratum": stratum_name,
                "tuples": [list(row) for row in instance.tuples],
                "realized_label": bool(instance.has_3sum),
                "construction_arm": (
                    "positive" if instance.construction_arm else "corrupted"
                ),
                "corruption_count": instance.corruption_count,
                "planted_matching_indices": (
                    list(instance.matching_indices)
                    if instance.matching_indices is not None
                    else None
                ),
                "near_match_2of3_count": feats["num_two_of_three_near_misses"],
                "candidate_coordinate_match_counts": feats["candidate_coordinate_match_counts"],
                "first_valid_witness_position": feats["first_valid_witness_position"],
            })

    provenance = {
        "structural_challenge_version": STRUCTURAL_CHALLENGE_VERSION,
        "challenge_id": spec.challenge_id,
        "spec": spec.canonical_dict(),
        "total_instances": len(ordered_records),
        "requested_strata": {name: wanted[name] for name in spec.strata},
        "realized_strata": {name: len(accepted[name]) for name in spec.strata},
        "attempts_per_stratum": {name: attempts[name] for name in spec.strata},
        "acceptance_rate_per_stratum": {
            name: len(accepted[name]) / attempts[name] if attempts[name] else None
            for name in spec.strata
        },
    }

    # Compute content SHA-256
    records_json = json.dumps(ordered_records, sort_keys=True, separators=(",", ":"))
    content_sha256 = hashlib.sha256(records_json.encode("utf-8")).hexdigest()
    provenance["content_sha256"] = content_sha256

    return {
        "provenance": provenance,
        "instances": ordered_records,
    }
