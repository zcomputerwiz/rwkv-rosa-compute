"""Deliberately rebalanced diagnostic challenge sets for Experiment 0.

Canonical validation samples the source-faithful distribution, in which
corrupted-arm instances that survive corruption and remain mathematically
positive are rare. That scarcity is the problem: it is the population where the
construction cue conflicts with the realized label, and there are too few of
them in canonical validation to measure anything per-N.

A challenge set suppresses the construction-arm prior by requesting a fixed
count per stratum. It is **not** a replacement for canonical validation and its
accuracy must never be averaged with it. Instances themselves are produced by
the unmodified source-faithful generator and are kept exactly as generated;
only which ones are retained differs, via rejection sampling.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from exp0.construction_strata import (
    CORRUPTED_ARM_NEGATIVE,
    CORRUPTED_ARM_SURVIVING_POSITIVE,
    POSITIVE_ARM_POSITIVE,
    diagnose_instance,
    primary_stratum,
)
from exp0.dataset import PackedInstances
from exp0.task3sum import (
    DEFAULT_CORRUPTION_RATE,
    SOURCE_GENERATOR,
    generate_instance,
)

# Bump when the acceptance rule or generation procedure changes, so a stored
# challenge set cannot be silently confused with one built under new rules.
CHALLENGE_SET_VERSION = 1

CHALLENGE_STRATA: Tuple[str, ...] = (
    POSITIVE_ARM_POSITIVE,
    CORRUPTED_ARM_NEGATIVE,
    CORRUPTED_ARM_SURVIVING_POSITIVE,
)

DEFAULT_MAX_ATTEMPTS_PER_STRATUM = 200_000


class ChallengeSetExhausted(RuntimeError):
    """A requested stratum could not be filled within its attempt budget."""


@dataclass(frozen=True)
class ChallengeSpec:
    """Everything that determines a challenge set's contents."""

    seed: int
    per_stratum: int = 1000
    length: int = 6
    dimension: int = 3
    mod: int = 10
    generator_mode: str = SOURCE_GENERATOR
    corruption_rate: float = DEFAULT_CORRUPTION_RATE
    max_attempts_per_stratum: int = DEFAULT_MAX_ATTEMPTS_PER_STRATUM
    strata: Tuple[str, ...] = CHALLENGE_STRATA

    def __post_init__(self):
        if self.per_stratum <= 0:
            raise ValueError("per_stratum must be positive")
        if self.max_attempts_per_stratum <= 0:
            raise ValueError("max_attempts_per_stratum must be positive")
        unknown = set(self.strata) - set(CHALLENGE_STRATA)
        if unknown:
            raise ValueError(f"unsupported challenge strata: {sorted(unknown)}")
        if not self.strata:
            raise ValueError("at least one stratum must be requested")

    def canonical_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["strata"] = sorted(self.strata)
        payload["challenge_set_version"] = CHALLENGE_SET_VERSION
        # The attempt budget bounds effort, not contents: two sets that differ
        # only by budget and both succeed are identical, so it is excluded from
        # identity.
        payload.pop("max_attempts_per_stratum", None)
        return payload

    @property
    def challenge_id(self) -> str:
        blob = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class ChallengeSet:
    instances: PackedInstances
    strata: List[str]
    provenance: Dict[str, Any]


def _generate_one(spec: ChallengeSpec, rng: random.Random, positive_arm: bool):
    return generate_instance(
        length=spec.length,
        dimension=spec.dimension,
        mod=spec.mod,
        target_has_3sum=positive_arm,
        rng=rng,
        generator_mode=spec.generator_mode,
        corruption_rate=spec.corruption_rate,
    )


def generate_challenge_set(spec: ChallengeSpec) -> ChallengeSet:
    """Build a stratum-balanced diagnostic set by rejection sampling.

    The positive arm is drawn for ``positive_arm_positive``; the corrupted arm
    supplies both ``corrupted_arm_negative`` and the rare surviving positives,
    so those two strata are filled from one stream and each draw is routed by
    its realized label.
    """
    wanted = {name: spec.per_stratum for name in spec.strata}
    accepted: Dict[str, List[Any]] = {name: [] for name in spec.strata}
    attempts: Dict[str, int] = {name: 0 for name in spec.strata}

    # Independent child streams keep one stratum's budget from perturbing
    # another's draws.
    root = random.Random(spec.seed)
    arm_streams = {
        "positive": random.Random(root.getrandbits(128)),
        "corrupted": random.Random(root.getrandbits(128)),
    }

    def needed(names) -> List[str]:
        return [n for n in names if n in wanted and len(accepted[n]) < wanted[n]]

    # Positive arm.
    positive_targets = needed([POSITIVE_ARM_POSITIVE])
    budget = spec.max_attempts_per_stratum
    while needed(positive_targets):
        if attempts[POSITIVE_ARM_POSITIVE] >= budget:
            raise ChallengeSetExhausted(
                f"{POSITIVE_ARM_POSITIVE}: filled "
                f"{len(accepted[POSITIVE_ARM_POSITIVE])}/{wanted[POSITIVE_ARM_POSITIVE]} "
                f"after {attempts[POSITIVE_ARM_POSITIVE]} attempts"
            )
        attempts[POSITIVE_ARM_POSITIVE] += 1
        instance = _generate_one(spec, arm_streams["positive"], positive_arm=True)
        if primary_stratum(diagnose_instance(instance, 0, mod=spec.mod)) == (
            POSITIVE_ARM_POSITIVE
        ):
            accepted[POSITIVE_ARM_POSITIVE].append(instance)

    # Corrupted arm feeds both corrupted strata from a single stream.
    corrupted_names = [
        name for name in (CORRUPTED_ARM_NEGATIVE, CORRUPTED_ARM_SURVIVING_POSITIVE)
        if name in wanted
    ]
    while needed(corrupted_names):
        for name in needed(corrupted_names):
            if attempts[name] >= spec.max_attempts_per_stratum:
                raise ChallengeSetExhausted(
                    f"{name}: filled {len(accepted[name])}/{wanted[name]} after "
                    f"{attempts[name]} attempts. Surviving positives are rare; "
                    "raise max_attempts_per_stratum or lower per_stratum."
                )
        instance = _generate_one(spec, arm_streams["corrupted"], positive_arm=False)
        stratum = primary_stratum(diagnose_instance(instance, 0, mod=spec.mod))
        # Every corrupted draw costs an attempt for each stratum still open,
        # which is what makes the acceptance rates below interpretable.
        for name in needed(corrupted_names):
            attempts[name] += 1
        if stratum in accepted and len(accepted[stratum]) < wanted[stratum]:
            accepted[stratum].append(instance)

    ordered_strata: List[str] = []
    ordered_instances: List[Any] = []
    for name in sorted(spec.strata):
        for instance in accepted[name]:
            ordered_strata.append(name)
            ordered_instances.append(instance)

    packed = _pack(ordered_instances, spec)
    provenance = {
        "challenge_set_version": CHALLENGE_SET_VERSION,
        "challenge_id": spec.challenge_id,
        "spec": spec.canonical_dict(),
        "max_attempts_per_stratum": spec.max_attempts_per_stratum,
        "requested_strata": {name: wanted[name] for name in sorted(wanted)},
        "realized_strata": {
            name: len(accepted[name]) for name in sorted(accepted)
        },
        "attempts_per_stratum": {name: attempts[name] for name in sorted(attempts)},
        "acceptance_rate_per_stratum": {
            name: (len(accepted[name]) / attempts[name] if attempts[name] else None)
            for name in sorted(accepted)
        },
        "total_attempts": sum(attempts.values()),
        "total_instances": len(ordered_instances),
    }
    return ChallengeSet(instances=packed, strata=ordered_strata, provenance=provenance)


def _pack(instances: List[Any], spec: ChallengeSpec) -> PackedInstances:
    count = len(instances)
    tuple_array = np.zeros((count, spec.length, spec.dimension), dtype=np.uint8)
    label_array = np.zeros(count, dtype=np.bool_)
    match_array = np.full((count, 3), -1, dtype=np.int16)
    arm_array = np.full(count, -1, dtype=np.int8)
    corruption_array = np.full(count, -1, dtype=np.int8)
    for idx, instance in enumerate(instances):
        tuple_array[idx] = instance.tuples
        label_array[idx] = instance.has_3sum
        if instance.matching_indices is not None:
            match_array[idx] = instance.matching_indices
        if instance.construction_arm is not None:
            arm_array[idx] = int(instance.construction_arm)
        if instance.corruption_count is not None:
            corruption_array[idx] = int(instance.corruption_count)
    return PackedInstances(
        tuples=torch.from_numpy(tuple_array),
        has_3sum=torch.from_numpy(label_array),
        matching_indices=torch.from_numpy(match_array),
        construction_arm=torch.from_numpy(arm_array),
        corruption_count=torch.from_numpy(corruption_array),
    )


def challenge_set_report(
    challenge: ChallengeSet,
    stratified: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the challenge-set section of a run report.

    Kept under its own key so it can never be mistaken for, or averaged with,
    canonical validation.
    """
    report: Dict[str, Any] = {
        "provenance": challenge.provenance,
        "distribution_note": (
            "Deliberately rebalanced diagnostic set. Accuracy here is NOT "
            "comparable to canonical filler_accuracy and must not replace it."
        ),
    }
    if stratified is not None:
        report["stratified"] = stratified
    return report
