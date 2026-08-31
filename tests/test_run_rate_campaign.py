"""The campaign driver's three pre-registered rules must actually hold.

Each of these exists to stop the campaign selecting on noise, and each is the
kind of rule that is easy to write into a protocol document and quietly not
implement.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_rate_campaign", REPO_ROOT / "scripts" / "run_rate_campaign.py")
campaign = importlib.util.module_from_spec(_spec)
sys.modules["run_rate_campaign"] = campaign
_spec.loader.exec_module(campaign)


def test_conditions_are_step_matched_as_pre_registered():
    """C1 and C2 must be step-matched to each other and 4x B0.

    The comparison the campaign makes is between two complete protocols at
    equal optimiser budget. If the step counts drift apart, C1 versus C2 stops
    being the contrast that was registered.
    """
    def steps(c):
        return (c["memories"] * 4 // 256) * c["epochs"]

    b0, c1, c2 = (steps(campaign.CONDITIONS[k]) for k in ("B0", "C1", "C2"))
    assert b0 == 9_984
    assert c1 == c2 == 39_936
    assert c1 == 4 * b0


def test_every_bank_is_cudagraph_aligned():
    """A ragged bank is refused by the audit tool, so catch it here instead."""
    for name, cfg in campaign.CONDITIONS.items():
        assert (cfg["memories"] * 4) % 256 == 0, f"{name} train bank is ragged"
    assert (448 * 4) % 256 == 0, "held-out bank is ragged"


def test_condition_order_rotates_so_it_is_not_confounded_with_time():
    """Each condition must appear in each slot equally across three seeds."""
    orders = [campaign.rotation(i) for i in range(3)]
    assert orders == [["B0", "C1", "C2"], ["C1", "C2", "B0"], ["C2", "B0", "C1"]]
    for slot in range(3):
        assert sorted(o[slot] for o in orders) == ["B0", "C1", "C2"]
    # And it repeats, so a ten-seed panel stays balanced rather than drifting.
    assert campaign.rotation(3) == campaign.rotation(0)


def test_the_panels_are_disjoint_and_avoid_the_gate_and_data_seeds():
    disc = set(campaign.PANELS["discovery"])
    conf = set(campaign.PANELS["confirmatory"])
    assert len(disc) == 10 and len(conf) == 20
    assert not disc & conf, "confirmatory seeds must not have chosen the winner"
    reserved = {1001, 1002, 1003, 1004, 1005, 4242}
    assert not (disc | conf) & reserved


def test_a_dirty_tree_aborts_rather_than_producing_incomparable_artifacts(
        tmp_path, monkeypatch):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "scratch.txt").write_text("uncommitted")
    monkeypatch.setattr(campaign, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="dirty"):
        campaign.frozen_tree()


def test_an_existing_artifact_is_never_overwritten(tmp_path, monkeypatch, capsys):
    """First valid run wins: a rerun must not replace a recorded observation."""
    # Every seed in the confirmatory panel already has a result, so a correct
    # driver launches nothing at all.
    for seed in campaign.PANELS["confirmatory"]:
        (tmp_path / f"ada_confirmatory_B0_s{seed}.json").write_text(
            '{"results": [{"held_out_final": 0.5}]}')

    def explode(*a, **k):
        raise AssertionError("the driver re-ran a seed that already had a result")

    monkeypatch.setattr(campaign.subprocess, "run", explode)
    monkeypatch.setattr(campaign, "frozen_tree", lambda: ("c" * 40, "t" * 40))
    campaign.main(["--node", "ada", "--panel", "confirmatory",
                   "--condition", "B0", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.count("skip") == 20
    assert "first valid run stands" in out


def test_the_confirmatory_panel_requires_a_named_winner():
    with pytest.raises(SystemExit):
        campaign.main(["--node", "ada", "--panel", "confirmatory"])
