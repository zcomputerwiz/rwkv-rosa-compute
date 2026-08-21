"""CPU tests for construction-prior diagnostics and challenge sets."""

import json
import random

import pytest

from exp0.challenge_set import (
    CHALLENGE_SET_VERSION,
    ChallengeSetExhausted,
    ChallengeSpec,
    generate_challenge_set,
)
from exp0.construction_strata import (
    CORRUPTED_ARM_NEGATIVE,
    CORRUPTED_ARM_SURVIVING_POSITIVE,
    POSITIVE_ARM_POSITIVE,
    EvaluationRecord,
    InstanceDiagnostics,
    build_records,
    compare_strata,
    corruption_stratum,
    diagnose_instance,
    diagnose_packed,
    enumerate_witnesses,
    error_records,
    fisher_exact_2x2,
    primary_stratum,
    summarize_strata,
)
from exp0.generation import generate_protocol_packed_instances
from exp0.task3sum import check_3sum, generate_instance

pytestmark = pytest.mark.exp0


def _diagnostics(**overrides) -> InstanceDiagnostics:
    base = dict(
        index=0,
        construction_arm="corrupted",
        realized_label=True,
        corruption_count=2,
        num_valid_triples=1,
        first_witness=(1, 3, 4),
        planted_matching_indices=(1, 3, 4),
        multiple_witnesses=False,
    )
    base.update(overrides)
    return InstanceDiagnostics(**base)


# --- structural classification ------------------------------------------------

def test_enumerate_witnesses_agrees_with_check_3sum_on_existence():
    """Enumeration is the only new Match-3 math; it must match the source rule."""
    rng = random.Random(4242)
    for _ in range(200):
        instance = generate_instance(length=6, dimension=3, rng=rng)
        tuples = [tuple(row) for row in instance.tuples]
        has_solution, first = check_3sum(tuples, mod=10)
        witnesses = enumerate_witnesses(tuples, mod=10)
        assert has_solution == bool(witnesses)
        if has_solution:
            assert first in witnesses


def test_enumerate_witnesses_counts_every_triple():
    # Three disjoint zero-sum triples in one instance, dimension 1.
    tuples = [(0,), (0,), (0,), (5,), (5,), (0,)]
    witnesses = enumerate_witnesses(tuples, mod=10)
    assert (0, 1, 2) in witnesses
    assert all(sum(tuples[i][0] for i in w) % 10 == 0 for w in witnesses)
    assert len(witnesses) == len(set(witnesses))


def test_multiple_witness_flag_tracks_the_count():
    assert not _diagnostics(num_valid_triples=1, multiple_witnesses=False).multiple_witnesses
    assert _diagnostics(num_valid_triples=3, multiple_witnesses=True).multiple_witnesses


# --- strata -------------------------------------------------------------------

def test_positive_arm_stratum():
    diagnostics = _diagnostics(construction_arm="positive", realized_label=True,
                               corruption_count=None)
    assert primary_stratum(diagnostics) == POSITIVE_ARM_POSITIVE
    # The positive arm has no corruption sub-stratum.
    assert corruption_stratum(diagnostics) is None


def test_corrupted_negative_stratum_and_subdivision():
    diagnostics = _diagnostics(realized_label=False, corruption_count=3,
                               num_valid_triples=0, first_witness=None,
                               planted_matching_indices=None)
    assert primary_stratum(diagnostics) == CORRUPTED_ARM_NEGATIVE
    assert corruption_stratum(diagnostics) == "corrupted_negative_c3"


def test_unknown_arm_is_not_silently_counted_as_positive():
    diagnostics = _diagnostics(construction_arm=None)
    assert primary_stratum(diagnostics) == "unknown_construction_arm"
    assert corruption_stratum(diagnostics) is None


@pytest.mark.parametrize("count", [1, 2, 3])
def test_corruption_count_subdivides_surviving_positives(count):
    diagnostics = _diagnostics(corruption_count=count)
    assert primary_stratum(diagnostics) == CORRUPTED_ARM_SURVIVING_POSITIVE
    assert corruption_stratum(diagnostics) == f"corrupted_positive_c{count}"


# --- the idx1743 regression ---------------------------------------------------

def test_idx1743_structure_is_a_corrupted_surviving_positive():
    """Regression for the structural category idx1743 exposed.

    This asserts categorization, NOT that a model should mispredict it.
    """
    diagnostics = _diagnostics(
        index=1743, construction_arm="corrupted", realized_label=True,
        corruption_count=2, num_valid_triples=1, first_witness=(1, 3, 4),
        planted_matching_indices=(1, 3, 4), multiple_witnesses=False,
    )
    assert primary_stratum(diagnostics) == CORRUPTED_ARM_SURVIVING_POSITIVE
    assert corruption_stratum(diagnostics) == "corrupted_positive_c2"
    assert diagnostics.prior_conflicts_with_label


def test_idx1743_regenerates_from_the_canonical_evaluation_seed():
    """The real instance, rebuilt from eval_seed 9999, has that structure."""
    packed = generate_protocol_packed_instances(
        num_samples=2000, length=6, dimension=3, mod=10, true_rate=0.5,
        rng=random.Random(9999), collect_provenance=True,
    )
    diagnostics = diagnose_instance(packed.instance_at(1743), 1743, mod=10)
    assert diagnostics.construction_arm == "corrupted"
    assert diagnostics.realized_label is True
    assert diagnostics.corruption_count == 2
    assert diagnostics.num_valid_triples == 1
    assert primary_stratum(diagnostics) == CORRUPTED_ARM_SURVIVING_POSITIVE
    assert corruption_stratum(diagnostics) == "corrupted_positive_c2"


def test_diagnose_packed_without_provenance_reports_unknown_arm():
    packed = generate_protocol_packed_instances(
        num_samples=8, length=6, dimension=3, rng=random.Random(3),
    )
    diagnostics = diagnose_packed(packed)
    assert all(d.construction_arm is None for d in diagnostics)
    assert all(primary_stratum(d) == "unknown_construction_arm" for d in diagnostics)


# --- summaries ----------------------------------------------------------------

def _record(correct: bool, **overrides) -> EvaluationRecord:
    diagnostics = _diagnostics(**overrides)
    predicted = diagnostics.realized_label if correct else not diagnostics.realized_label
    return EvaluationRecord(diagnostics=diagnostics, predicted_label=predicted,
                            correct=correct, true_logit=1.0, false_logit=0.25)


def test_summary_reports_counts_confusion_and_accuracy():
    summary = summarize_strata([
        _record(True, index=0),
        _record(False, index=1),
        _record(True, index=2, construction_arm="positive", corruption_count=None),
    ])
    surviving = summary["construction_strata"][CORRUPTED_ARM_SURVIVING_POSITIVE]
    assert surviving["count"] == 2
    assert surviving["errors"] == 1
    assert surviving["accuracy"] == 0.5
    assert surviving["true_positive"] == 1
    assert surviving["false_negative"] == 1
    assert summary["overall"]["count"] == 3


def test_empty_stratum_reports_null_accuracy_not_zero():
    summary = summarize_strata([_record(True)])
    empty = summary["construction_strata"][POSITIVE_ARM_POSITIVE]
    assert empty["count"] == 0
    assert empty["errors"] == 0
    # 0.0 would read as total failure rather than absence of data.
    assert empty["accuracy"] is None
    assert summary["corruption_strata"]["corrupted_positive_c3"]["accuracy"] is None


def test_every_known_stratum_is_present_for_sweep_alignment():
    summary = summarize_strata([])
    assert len(summary["construction_strata"]) == 5
    assert len(summary["corruption_strata"]) == 6
    assert summary["overall"]["accuracy"] is None


def test_records_and_summary_are_json_serializable():
    summary = summarize_strata([_record(True), _record(False, index=1)])
    payload = {"stratified": summary, "errors": error_records(
        [_record(False, index=5), _record(False, index=2)]
    )}
    decoded = json.loads(json.dumps(payload))
    assert decoded["stratified"]["overall"]["count"] == 2
    assert decoded["errors"][0]["primary_stratum"] == CORRUPTED_ARM_SURVIVING_POSITIVE


def test_error_records_are_ordered_by_validation_index():
    records = [_record(False, index=idx) for idx in (7, 1, 5, 0)]
    assert [row["index"] for row in error_records(records)] == [0, 1, 5, 7]


def test_error_records_carry_margin_when_logits_were_captured():
    row = error_records([_record(False, index=0)])[0]
    assert row["true_logit"] == 1.0
    assert row["prediction_margin"] == pytest.approx(0.75)


def test_build_records_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_records([_diagnostics()], [True, False])


def test_unparseable_prediction_counts_as_an_error():
    record = build_records([_diagnostics()], [None])[0]
    assert record.correct is False
    assert record.prediction_margin is None


# --- statistics ---------------------------------------------------------------

def test_fisher_matches_known_two_sided_value():
    # Classic tea-tasting table; the exact two-sided p is 1.0 by symmetry.
    assert fisher_exact_2x2(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-3)
    assert fisher_exact_2x2(0, 0, 0, 0) == 1.0
    with pytest.raises(ValueError):
        fisher_exact_2x2(-1, 1, 1, 1)


def test_compare_strata_flags_small_subgroups():
    records = [_record(True, index=i) for i in range(10)]
    records.append(_record(False, index=99, construction_arm="positive",
                           corruption_count=None))
    summary = summarize_strata(records)
    comparison = compare_strata(summary, POSITIVE_ARM_POSITIVE,
                                CORRUPTED_ARM_SURVIVING_POSITIVE).to_dict()
    assert comparison["left_count"] == 1
    assert comparison["right_count"] == 10
    assert "small stratum" in comparison["note"]
    assert 0.0 <= comparison["p_value"] <= 1.0


def test_compare_strata_skips_the_test_when_a_stratum_is_empty():
    summary = summarize_strata([_record(True)])
    comparison = compare_strata(summary, POSITIVE_ARM_POSITIVE,
                                CORRUPTED_ARM_SURVIVING_POSITIVE).to_dict()
    assert comparison["p_value"] is None
    assert "empty" in comparison["note"]


# --- challenge sets -----------------------------------------------------------

def test_challenge_set_fills_every_requested_stratum():
    challenge = generate_challenge_set(
        ChallengeSpec(seed=11, per_stratum=12, length=6, dimension=3)
    )
    realized = challenge.provenance["realized_strata"]
    assert realized == {name: 12 for name in realized}
    assert len(challenge.strata) == 36
    diagnostics = diagnose_packed(challenge.instances)
    observed = [primary_stratum(d) for d in diagnostics]
    assert observed == challenge.strata


def test_challenge_set_rejection_sampling_is_visible_in_provenance():
    challenge = generate_challenge_set(
        ChallengeSpec(seed=5, per_stratum=10, length=6, dimension=3)
    )
    attempts = challenge.provenance["attempts_per_stratum"]
    rates = challenge.provenance["acceptance_rate_per_stratum"]
    # Surviving positives are rare, so they must cost more attempts than the
    # positive arm, which is accepted on every draw.
    assert attempts[CORRUPTED_ARM_SURVIVING_POSITIVE] > attempts[POSITIVE_ARM_POSITIVE]
    assert rates[POSITIVE_ARM_POSITIVE] == 1.0
    assert rates[CORRUPTED_ARM_SURVIVING_POSITIVE] < 1.0
    assert challenge.provenance["total_attempts"] >= sum(
        challenge.provenance["realized_strata"].values()
    )


def test_challenge_set_is_deterministic_for_a_seed():
    first = generate_challenge_set(ChallengeSpec(seed=99, per_stratum=8, length=6))
    second = generate_challenge_set(ChallengeSpec(seed=99, per_stratum=8, length=6))
    assert first.instances.tuples.equal(second.instances.tuples)
    assert first.strata == second.strata
    assert first.provenance["attempts_per_stratum"] == (
        second.provenance["attempts_per_stratum"]
    )
    other = generate_challenge_set(ChallengeSpec(seed=100, per_stratum=8, length=6))
    assert not first.instances.tuples.equal(other.instances.tuples)


def test_challenge_id_is_deterministic_and_specification_sensitive():
    base = ChallengeSpec(seed=7, per_stratum=50, length=6)
    assert base.challenge_id == ChallengeSpec(seed=7, per_stratum=50, length=6).challenge_id
    assert base.challenge_id != ChallengeSpec(seed=8, per_stratum=50, length=6).challenge_id
    assert base.challenge_id != ChallengeSpec(seed=7, per_stratum=51, length=6).challenge_id
    # Attempt budget bounds effort, not contents, so it is not part of identity.
    relaxed = ChallengeSpec(seed=7, per_stratum=50, length=6,
                            max_attempts_per_stratum=123)
    assert relaxed.challenge_id == base.challenge_id


def test_challenge_provenance_records_reproduction_metadata():
    spec = ChallengeSpec(seed=3, per_stratum=6, length=6, dimension=3)
    provenance = generate_challenge_set(spec).provenance
    assert provenance["challenge_set_version"] == CHALLENGE_SET_VERSION
    assert provenance["challenge_id"] == spec.challenge_id
    for key in ("length", "dimension", "mod", "generator_mode", "seed"):
        assert key in provenance["spec"]
    assert provenance["total_instances"] == 18
    json.dumps(provenance)  # provenance must survive report serialization


def test_challenge_set_fails_clearly_when_a_stratum_cannot_be_filled():
    with pytest.raises(ChallengeSetExhausted, match="corrupted_arm_surviving_positive"):
        generate_challenge_set(ChallengeSpec(
            seed=1, per_stratum=5, length=6,
            strata=(CORRUPTED_ARM_SURVIVING_POSITIVE,),
            max_attempts_per_stratum=3,
        ))


def test_challenge_spec_rejects_invalid_requests():
    with pytest.raises(ValueError):
        ChallengeSpec(seed=1, per_stratum=0)
    with pytest.raises(ValueError):
        ChallengeSpec(seed=1, strata=("not_a_stratum",))
    with pytest.raises(ValueError):
        ChallengeSpec(seed=1, max_attempts_per_stratum=0)


# --- report integration -------------------------------------------------------

def _report(per_seed_results):
    from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
    from exp0.evaluate import compile_experiment_report

    return compile_experiment_report(
        ModelConfig(), TrainConfig(), Task3SumConfig(), per_seed_results,
        majority_class_baseline=0.5,
        realized_mixture_counts={"parallel_cot": 1, "filler": 1},
        eval_seed=123, val_samples=10,
    )


def _seed_result(**extra):
    base = {
        "seed": 42, "task_seed": 42, "training_seed": 42,
        "best_filler_accuracy": 0.9,
        "best_online_train_answer_accuracy": 0.9,
    }
    base.update(extra)
    return [base]


def test_report_omits_diagnostics_when_they_were_not_requested():
    """A run without the flag must be byte-identical to one before this change."""
    assert "construction_diagnostics" not in _report(_seed_result())


def test_report_keeps_canonical_and_challenge_diagnostics_separate():
    report = _report(_seed_result(
        construction_diagnostics={"distribution": "canonical_source_faithful"},
        challenge_diagnostics={"provenance": {"challenge_id": "abc"}},
    ))
    diagnostics = report["construction_diagnostics"]
    assert set(diagnostics) == {
        "canonical_validation", "diagnostic_challenge_validation"
    }
    assert diagnostics["canonical_validation"][0]["distribution"] == (
        "canonical_source_faithful"
    )
    assert diagnostics["diagnostic_challenge_validation"][0]["provenance"][
        "challenge_id"
    ] == "abc"


def test_detail_sink_is_filled_by_the_existing_validation_pass():
    """Per-example detail must come from the canonical pass, not a second one."""
    import torch
    from torch.utils.data import DataLoader

    from exp0.config import ModelConfig, Task3SumConfig
    from exp0.dataset import Task3SumDataset, build_default_vocab, pad_collate_fn
    from exp0.train import create_model, evaluate_accuracy

    task_cfg = Task3SumConfig(length=6, dimension=3, num_filler=0, num_samples=8)
    vocab = build_default_vocab(length=6, dimension=3)
    packed = generate_protocol_packed_instances(
        num_samples=8, length=6, dimension=3, rng=random.Random(17),
        collect_provenance=True,
    )
    dataset = Task3SumDataset(packed, format_type="filler", num_filler=0,
                              vocab=vocab)
    model = create_model(
        ModelConfig(architecture="llama", hidden_size=32, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=64, device="cpu",
                    vocab_size=len(vocab)),
        d_input=10 * 3 + 6, vocab=vocab, task_cfg=task_cfg,
    )
    sink = {}
    accuracy = evaluate_accuracy(
        model, DataLoader(dataset, batch_size=4, collate_fn=pad_collate_fn),
        torch.device("cpu"), vocab.token2id["ANS"], vocab.token2id["True"],
        vocab.token2id["False"], detail_sink=sink,
    )
    assert 0.0 <= accuracy <= 1.0
    assert len(sink["predicted_ids"]) == 8
    assert len(sink["true_logits"]) == len(sink["false_logits"]) == 8
    assert len(sink["labels"]) == 8

    records = build_records(
        diagnose_packed(packed), [None] * 8, sink["true_logits"],
        sink["false_logits"],
    )
    assert summarize_strata(records)["overall"]["count"] == 8
