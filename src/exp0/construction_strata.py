"""Construction-prior diagnostics for Experiment 0 validation.

The source-faithful generator picks a *construction arm* before it builds an
instance: it either plants a Match-3 solution, or plants one and then corrupts
it. The corrupted arm is strongly associated with the negative label but does
not determine it, because corruption can fail to destroy the planted solution.
Those survivors are exactly the examples where the generation cue conflicts with
the realized mathematical label, and they are the population this module makes
addressable.

Nothing here changes the canonical validation distribution or metric. These are
supplementary diagnostics computed from metadata the generator already records,
plus ``check_3sum`` for the source-faithful first witness.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from exp0.task3sum import check_3sum

# Primary strata. The third is the point of this module.
POSITIVE_ARM_POSITIVE = "positive_arm_positive"
CORRUPTED_ARM_NEGATIVE = "corrupted_arm_negative"
CORRUPTED_ARM_SURVIVING_POSITIVE = "corrupted_arm_surviving_positive"
# A positive-arm instance that is somehow negative would mean the planted
# solution was lost, which generate_source_instance refuses to produce. It is
# named so an impossible case surfaces as a stratum rather than a crash.
POSITIVE_ARM_NEGATIVE = "positive_arm_negative"
UNKNOWN_ARM = "unknown_construction_arm"

PRIMARY_STRATA: Tuple[str, ...] = (
    POSITIVE_ARM_POSITIVE,
    CORRUPTED_ARM_NEGATIVE,
    CORRUPTED_ARM_SURVIVING_POSITIVE,
    POSITIVE_ARM_NEGATIVE,
    UNKNOWN_ARM,
)

MAX_CORRUPTIONS = 3
CORRUPTION_STRATA: Tuple[str, ...] = tuple(
    f"corrupted_{polarity}_c{count}"
    for polarity in ("positive", "negative")
    for count in range(1, MAX_CORRUPTIONS + 1)
)


@dataclass(frozen=True)
class InstanceDiagnostics:
    """Structural description of one validation instance."""

    index: int
    construction_arm: Optional[str]
    realized_label: bool
    corruption_count: Optional[int]
    num_valid_triples: int
    first_witness: Optional[Tuple[int, int, int]]
    planted_matching_indices: Optional[Tuple[int, int, int]]
    multiple_witnesses: bool

    @property
    def prior_conflicts_with_label(self) -> bool:
        """True when the construction cue points away from the real label."""
        return self.construction_arm == "corrupted" and self.realized_label


def enumerate_witnesses(
    tuples: Sequence[Sequence[int]],
    mod: int = 10,
) -> List[Tuple[int, int, int]]:
    """Every ``i < j < k`` whose tuples sum to zero in all dimensions.

    ``check_3sum`` returns only the first witness under the source's search
    order, so counting requires enumeration. The membership test is the same
    condition ``check_3sum`` applies, and a test asserts the two agree on which
    instances have any witness at all.
    """
    if not tuples:
        return []
    dimension = len(tuples[0])
    found: List[Tuple[int, int, int]] = []
    for i, j, k in itertools.combinations(range(len(tuples)), 3):
        if all(
            (tuples[i][d] + tuples[j][d] + tuples[k][d]) % mod == 0
            for d in range(dimension)
        ):
            found.append((i, j, k))
    return found


def diagnose_instance(instance, index: int, mod: int = 10) -> InstanceDiagnostics:
    """Classify one ``Instance3Sum`` using recorded provenance and check_3sum."""
    tuples = [tuple(row) for row in instance.tuples]
    witnesses = enumerate_witnesses(tuples, mod=mod)
    # Source-faithful ordering for the reported witness; enumeration only counts.
    has_solution, first_witness = check_3sum(tuples, mod=mod)
    arm = instance.construction_arm
    return InstanceDiagnostics(
        index=index,
        construction_arm=(
            None if arm is None else ("positive" if arm else "corrupted")
        ),
        realized_label=bool(instance.has_3sum),
        corruption_count=instance.corruption_count,
        num_valid_triples=len(witnesses),
        first_witness=tuple(first_witness) if has_solution and first_witness else None,
        planted_matching_indices=(
            tuple(instance.matching_indices)
            if instance.matching_indices is not None
            else None
        ),
        multiple_witnesses=len(witnesses) > 1,
    )


def diagnose_packed(packed, mod: int = 10) -> List[InstanceDiagnostics]:
    """Classify every instance in a ``PackedInstances`` collection.

    The collection must have been generated with ``collect_provenance=True``;
    without it the construction arm is unknown and every instance lands in the
    ``unknown_construction_arm`` stratum rather than being silently miscounted.
    """
    return [
        diagnose_instance(packed.instance_at(idx), idx, mod=mod)
        for idx in range(len(packed))
    ]


def primary_stratum(diagnostics: InstanceDiagnostics) -> str:
    if diagnostics.construction_arm is None:
        return UNKNOWN_ARM
    if diagnostics.construction_arm == "positive":
        return POSITIVE_ARM_POSITIVE if diagnostics.realized_label else (
            POSITIVE_ARM_NEGATIVE
        )
    return (
        CORRUPTED_ARM_SURVIVING_POSITIVE
        if diagnostics.realized_label
        else CORRUPTED_ARM_NEGATIVE
    )


def corruption_stratum(diagnostics: InstanceDiagnostics) -> Optional[str]:
    """Corruption-count sub-stratum, or None when the arm was not corrupted."""
    if diagnostics.construction_arm != "corrupted":
        return None
    if diagnostics.corruption_count is None:
        return None
    polarity = "positive" if diagnostics.realized_label else "negative"
    return f"corrupted_{polarity}_c{diagnostics.corruption_count}"


@dataclass
class EvaluationRecord:
    """One validation example plus how the model scored it."""

    diagnostics: InstanceDiagnostics
    predicted_label: Optional[bool]
    correct: bool
    true_logit: Optional[float] = None
    false_logit: Optional[float] = None

    @property
    def prediction_margin(self) -> Optional[float]:
        """Signed logit margin toward True; None when logits were not kept."""
        if self.true_logit is None or self.false_logit is None:
            return None
        return self.true_logit - self.false_logit

    def to_error_record(self) -> Dict[str, Any]:
        diagnostics = asdict(self.diagnostics)
        diagnostics.update({
            "predicted_label": self.predicted_label,
            "primary_stratum": primary_stratum(self.diagnostics),
            "corruption_stratum": corruption_stratum(self.diagnostics),
            "true_logit": self.true_logit,
            "false_logit": self.false_logit,
            "prediction_margin": self.prediction_margin,
        })
        return diagnostics


def _empty_summary() -> Dict[str, Any]:
    return {
        "count": 0,
        "correct": 0,
        "errors": 0,
        # An empty stratum has no accuracy. Reporting 0.0 would read as a
        # catastrophic failure rather than an absence of data.
        "accuracy": None,
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }


def _accumulate(summary: Dict[str, Any], record: EvaluationRecord) -> None:
    summary["count"] += 1
    if record.correct:
        summary["correct"] += 1
    else:
        summary["errors"] += 1
    label = record.diagnostics.realized_label
    predicted = record.predicted_label
    if predicted is None:
        return
    if label and predicted:
        summary["true_positive"] += 1
    elif label and not predicted:
        summary["false_negative"] += 1
    elif not label and predicted:
        summary["false_positive"] += 1
    else:
        summary["true_negative"] += 1


def summarize_strata(records: Iterable[EvaluationRecord]) -> Dict[str, Any]:
    """Per-stratum counts, error counts, accuracy, and confusion cells.

    Every known stratum is present even when empty, so a sweep can align
    columns across N values without inventing keys.
    """
    construction = {name: _empty_summary() for name in PRIMARY_STRATA}
    corruption = {name: _empty_summary() for name in CORRUPTION_STRATA}
    overall = _empty_summary()

    for record in records:
        _accumulate(overall, record)
        _accumulate(construction[primary_stratum(record.diagnostics)], record)
        sub = corruption_stratum(record.diagnostics)
        if sub is not None:
            _accumulate(corruption[sub], record)

    for summary in (overall, *construction.values(), *corruption.values()):
        if summary["count"]:
            summary["accuracy"] = summary["correct"] / summary["count"]

    return {
        "overall": overall,
        "construction_strata": construction,
        "corruption_strata": corruption,
    }


def build_records(
    diagnostics: Sequence[InstanceDiagnostics],
    predicted_labels: Sequence[Optional[bool]],
    true_logits: Optional[Sequence[float]] = None,
    false_logits: Optional[Sequence[float]] = None,
) -> List[EvaluationRecord]:
    if len(diagnostics) != len(predicted_labels):
        raise ValueError("diagnostics and predictions must be the same length")
    records = []
    for position, diagnostic in enumerate(diagnostics):
        predicted = predicted_labels[position]
        records.append(EvaluationRecord(
            diagnostics=diagnostic,
            predicted_label=predicted,
            correct=predicted is not None and predicted == diagnostic.realized_label,
            true_logit=None if true_logits is None else float(true_logits[position]),
            false_logit=None if false_logits is None else float(false_logits[position]),
        ))
    return records


def error_records(records: Iterable[EvaluationRecord]) -> List[Dict[str, Any]]:
    """Failed examples only, in ascending validation index order."""
    failures = [record for record in records if not record.correct]
    failures.sort(key=lambda record: record.diagnostics.index)
    return [record.to_error_record() for record in failures]


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for ``[[a, b], [c, d]]``.

    Implemented directly rather than adding SciPy, which is not a project
    dependency. Exact, not an approximation, and only sane for small tables.
    """
    for value in (a, b, c, d):
        if value < 0:
            raise ValueError("contingency counts must be non-negative")
    total = a + b + c + d
    if total == 0 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
        return 1.0

    def probability(x: int) -> float:
        return (
            math.comb(a + b, x)
            * math.comb(c + d, a + c - x)
            / math.comb(total, a + c)
        )

    observed = probability(a)
    low = max(0, a + c - (c + d))
    high = min(a + b, a + c)
    # Sum every table at most as likely as the observed one.
    tail = sum(
        probability(x)
        for x in range(low, high + 1)
        if probability(x) <= observed + 1e-12
    )
    return min(1.0, tail)


@dataclass
class StratumComparison:
    left: str
    right: str
    left_errors: int
    left_count: int
    right_errors: int
    right_count: int
    p_value: Optional[float] = None
    note: str = field(default="")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "left_errors": self.left_errors,
            "left_count": self.left_count,
            "left_error_rate": (
                self.left_errors / self.left_count if self.left_count else None
            ),
            "right_errors": self.right_errors,
            "right_count": self.right_count,
            "right_error_rate": (
                self.right_errors / self.right_count if self.right_count else None
            ),
            "p_value": self.p_value,
            "note": self.note,
        }


def compare_strata(
    summary: Dict[str, Any],
    left: str,
    right: str,
) -> StratumComparison:
    """Compare two strata's error rates with an exact test.

    Reports counts and rates whether or not the test is meaningful. A small
    subgroup with a large percentage is not evidence, and the returned note
    says so rather than leaving the caller to guess.
    """
    groups = {**summary["construction_strata"], **summary["corruption_strata"]}
    for name in (left, right):
        if name not in groups:
            raise KeyError(f"unknown stratum: {name}")
    left_summary, right_summary = groups[left], groups[right]
    comparison = StratumComparison(
        left=left,
        right=right,
        left_errors=left_summary["errors"],
        left_count=left_summary["count"],
        right_errors=right_summary["errors"],
        right_count=right_summary["count"],
    )
    if not left_summary["count"] or not right_summary["count"]:
        comparison.note = "at least one stratum is empty; no test performed"
        return comparison
    comparison.p_value = fisher_exact_2x2(
        left_summary["errors"],
        left_summary["count"] - left_summary["errors"],
        right_summary["errors"],
        right_summary["count"] - right_summary["errors"],
    )
    if min(left_summary["count"], right_summary["count"]) < 30:
        comparison.note = (
            "small stratum; treat the rate difference as descriptive only"
        )
    return comparison
