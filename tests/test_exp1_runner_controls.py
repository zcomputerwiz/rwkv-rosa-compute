"""The gate ladder needs the runner to separate what --seed used to fuse.

Gate 0c fixes the data and varies only the model, and evaluates on the training
set itself. Neither is expressible when one seed drives the train bank, the
held-out bank, the initialization, and the training RNG at once. These tests
pin the separation and the reporting contract the gates depend on.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_exp1_pointer_chase.py"

BASE = [
    "--depth", "1",
    "--num-nodes", "8",
    "--num-maps", "2",
    "--d-model", "64",
    "--layers", "1",
    "--precision", "fp32",
    "--batch-size", "8",
    "--epochs", "1",
    "--train-size", "4",
    "--val-size", "2",
    "--device", "cpu",
]


def raw(out: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(RUNNER), *BASE, "--out-dir", str(out), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def run_into(out: Path, *extra: str) -> None:
    proc = raw(out, *extra)
    assert proc.returncode == 0, proc.stderr


def run(tmp_path: Path, *extra: str) -> dict:
    out = tmp_path / "out"
    proc = raw(out, *extra)
    assert proc.returncode == 0, proc.stderr
    reports = list(out.glob("report_model_seed*.json"))
    assert len(reports) == 1, reports
    return json.loads(reports[0].read_text())


def test_seed_alone_still_drives_everything(tmp_path):
    """The split flags default back to the old behaviour."""
    r = run(tmp_path, "--seed", "7")
    c = r["config"]
    assert c["model_seed"] == 7
    assert c["train_data_seed"] == 7
    assert c["val_data_seed"] == 8  # the old seed + 1


def test_data_seeds_are_independent_of_the_model_seed(tmp_path):
    """Gate 0c's requirement: one fixed bank, three model initializations."""
    r = run(tmp_path, "--seed", "7", "--model-seed", "1001",
            "--train-data-seed", "1004", "--val-data-seed", "1005")
    c = r["config"]
    assert (c["model_seed"], c["train_data_seed"], c["val_data_seed"]) == (1001, 1004, 1005)


def test_the_split_seeds_actually_route_where_they_claim(tmp_path):
    """The report echoing its own arguments proves nothing.

    Reaching the same three sub-seeds by a different route must reproduce the
    run bit-for-bit. On CPU fp32 this path is deterministic, so an equal train
    loss is evidence the seeds reach the data and the initializer, not just the
    JSON.
    """
    direct = run(tmp_path / "a", "--seed", "7")
    routed = run(tmp_path / "b", "--seed", "99", "--model-seed", "7",
                 "--train-data-seed", "7", "--val-data-seed", "8")
    assert routed["history"]["epoch_train_losses"] == direct["history"]["epoch_train_losses"]
    assert routed["final_accuracy"] == direct["final_accuracy"]


def test_changing_only_the_train_data_seed_changes_the_run(tmp_path):
    """And the data seed must not be inert."""
    a = run(tmp_path / "a", "--seed", "7", "--train-data-seed", "1004")
    b = run(tmp_path / "b", "--seed", "7", "--train-data-seed", "1005")
    assert a["history"]["epoch_train_losses"] != b["history"]["epoch_train_losses"]


def test_instance_count_is_reported_not_just_memory_count(tmp_path):
    """--train-size is memories; a budget written in instances is off by 4x."""
    r = run(tmp_path, "--seed", "7", "--queries-per-memory", "4")
    c = r["config"]
    assert c["train_memories"] == 4
    assert c["train_instances"] == 16
    assert c["val_memories"] == 2
    assert c["val_instances"] == 8


def test_queries_per_memory_is_settable(tmp_path):
    r = run(tmp_path, "--seed", "7", "--queries-per-memory", "1")
    assert r["config"]["train_instances"] == 4


def test_kernel_is_explicit_and_defaults_to_reference_on_cpu(tmp_path):
    r = run(tmp_path, "--seed", "7")
    assert r["config"]["rwkv_kernel"] == "reference"


def test_qwen4_only_flags_are_not_silently_ignored(tmp_path):
    proc = raw(tmp_path, "--seed", "7", "--qwen4-variant", "hybrid")
    assert proc.returncode != 0
    assert "requires --architecture qwen4_exp" in proc.stderr

    proc = raw(
        tmp_path,
        "--seed", "7",
        "--architecture", "qwen4_exp",
        "--rwkv-kernel", "reference",
    )
    assert proc.returncode != 0
    assert "valid only with --architecture rwkv" in proc.stderr

    proc = raw(
        tmp_path,
        "--seed", "7",
        "--architecture", "qwen4_exp",
        "--compile",
    )
    assert proc.returncode != 0
    assert "not part of the registered pilot" in proc.stderr

    proc = raw(
        tmp_path,
        "--seed", "7",
        "--architecture", "qwen4_exp",
        "--d-model", "128",
        "--layers", "4",
        "--precision", "bf16",
    )
    assert proc.returncode != 0
    assert "requires --precision fp32" in proc.stderr


def test_qwen4_report_carries_the_resolved_architecture(tmp_path):
    report = run(
        tmp_path,
        "--seed", "7",
        "--architecture", "qwen4_exp",
        "--qwen4-variant", "all-gdn",
        "--d-model", "128",
        "--layers", "4",
        "--train-size", "1",
        "--val-size", "1",
        "--queries-per-memory", "1",
        "--batch-size", "1",
    )
    config = report["config"]
    assert config["architecture"] == "qwen4_exp"
    assert config["rwkv_kernel"] is None
    assert config["qwen4_variant"] == "all-gdn"
    assert config["qwen4_config"]["transformers_version"] == "5.16.0"
    assert config["qwen4_config"]["layer_types"] == ["linear_attention"] * 4
    assert config["trainable_parameters"] > 0
    r = run(tmp_path / "b", "--seed", "7", "--rwkv-kernel", "reference")
    assert r["config"]["rwkv_kernel"] == "reference"


def test_gate_runner_defaults_to_no_workspace(tmp_path):
    c = run(tmp_path, "--seed", "7")["config"]
    assert c["workspace"] is False
    assert (c["num_slots"], c["num_steps"], c["m_max"]) == (None, None, None)


def test_workspace_is_explicit_and_fully_reported(tmp_path):
    c = run(
        tmp_path, "--seed", "7", "--workspace",
        "--num-slots", "2", "--num-steps", "3", "--m-max", "8",
    )["config"]
    assert c["workspace"] is True
    assert (c["num_slots"], c["num_steps"], c["m_max"]) == (2, 3, 8)


def test_workspace_dimensions_cannot_be_silently_ignored(tmp_path):
    proc = raw(tmp_path, "--seed", "7", "--num-slots", "8")
    assert proc.returncode != 0
    assert "require --workspace" in proc.stderr


def test_overfit_mode_evaluates_on_the_training_set(tmp_path):
    r = run(tmp_path, "--seed", "7", "--overfit-train-as-val")
    assert r["eval_target"] == "train_set"
    assert r["config"]["overfit_train_as_val"] is True
    # The held-out bank is still scored, as a separate non-gating number.
    assert r["holdout_diagnostic_accuracy"] is not None


def test_holdout_diagnostic_is_absent_when_not_overfitting(tmp_path):
    r = run(tmp_path, "--seed", "7")
    assert r["eval_target"] == "held_out"
    assert r["holdout_diagnostic_accuracy"] is None


def test_reported_outcome_is_the_final_epoch_not_the_best(tmp_path):
    """Fixed budget, no checkpoint selection: the outcome is the last epoch."""
    r = run(tmp_path, "--seed", "7", "--epochs", "2")
    assert r["final_accuracy"] == r["epoch_accuracies"][-1]
    assert len(r["epoch_accuracies"]) == 2


def test_report_is_named_for_the_model_seed(tmp_path):
    """Three gate attempts vary --model-seed with --seed held equal."""
    out = tmp_path / "out"
    for ms in ("1001", "1002", "1003"):
        run_into(out, "--seed", "7", "--model-seed", ms)
    assert sorted(p.name for p in out.glob("*.json")) == [
        "report_model_seed1001.json",
        "report_model_seed1002.json",
        "report_model_seed1003.json",
    ]


def test_an_existing_report_is_never_overwritten(tmp_path):
    out = tmp_path / "out"
    run_into(out, "--seed", "7")
    proc = raw(out, "--seed", "7")
    assert proc.returncode != 0
    assert "refusing to overwrite" in proc.stderr
