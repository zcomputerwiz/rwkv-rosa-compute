import json

# We need to import the script logic.
# Because it's in scripts/ we can import it if we mock or handle it.
import sys
from pathlib import Path

# Add the project root to sys.path if not there
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.audit_word_conditioned_floor import estimate_floor, main


def test_uniform_word_yields_chance():
    # A word where a selector appears exactly once and others don't matter or
    # it's a short depth that doesn't concentrate.
    # At depth 1, composition of permutations is exactly a single permutation,
    # so starting uniform -> ending uniform. P(answer == s) = 1/M.
    M = 16
    K = 4
    words = 20
    memories = 50
    bootstrap = 100
    seed = 42

    e_bw, bound = estimate_floor(
        num_nodes=M, num_maps=K, words=words, memories=memories, bootstrap=bootstrap, seed=seed, depth=1
    )

    # E[Bw] should be very close to 1/M = 0.0625
    assert abs(e_bw - 1.0 / M) < 0.015

def test_concentrated_word_yields_above_chance():
    """An all-identical word is elevated, and by a predictable amount.

    Permutations do not concentrate onto attractors the way arbitrary maps do,
    so the elevation here is not an attractor effect. It comes from cycle
    structure: for w = (0,) * D the composition is pi^D, and a node returns to
    itself exactly when its cycle length divides D. The expected fixed-point
    count is therefore

        E[fix(pi^D)] = #{ l <= M : l divides D }

    restricted to l <= M, because a cycle cannot be longer than the ground set.
    At M=16, D=16 the divisors 1, 2, 4, 8, 16 all qualify, giving 5 expected
    fixed points and p_w = 5/16 = 0.3125 -- five times chance.
    """
    M = 16
    K = 4
    words = 1
    memories = 100
    bootstrap = 0
    seed = 42

    # Let's pass a specific word:
    D = 16
    # An all-identical word, e.g., map 0 repeated 16 times
    word = (0,) * D

    e_bw, _ = estimate_floor(
        num_nodes=M, num_maps=K, words=words, memories=memories, bootstrap=bootstrap, seed=seed, word_list=[word]
    )

    # Predicted 5/16 from the divisor count above; asserted loosely because
    # the estimator scores on a finite second bank.
    assert e_bw > 1.0 / M + 0.05
    assert abs(e_bw - 5.0 / M) < 0.05, (
        f"expected about {5.0 / M:.4f} from #{{l <= 16 : l | 16}} = 5, got {e_bw:.4f}")

def test_reproducibility():
    M = 16
    K = 4
    words = 10
    memories = 10
    bootstrap = 10
    seed = 123
    depth = 4

    e1, b1 = estimate_floor(
        num_nodes=M, num_maps=K, words=words, memories=memories, bootstrap=bootstrap, seed=seed, depth=depth
    )
    e2, b2 = estimate_floor(
        num_nodes=M, num_maps=K, words=words, memories=memories, bootstrap=bootstrap, seed=seed, depth=depth
    )

    assert e1 == e2
    assert b1 == b2

def test_cli_integration(monkeypatch, tmp_path):
    out_file = tmp_path / "result.json"

    test_args = [
        "audit_word_conditioned_floor.py",
        "--depths", "2", "4",
        "--num-nodes", "16",
        "--num-maps", "4",
        "--words", "10",
        "--memories", "10",
        "--bootstrap", "10",
        "--seed", "999",
        "--out", str(out_file)
    ]

    monkeypatch.setattr("sys.argv", test_args)
    main()

    assert out_file.exists()

    with open(out_file, "r") as f:
        data = json.load(f)

    assert "parameters" in data
    assert "results" in data

    assert data["parameters"]["depths"] == [2, 4]
    assert data["parameters"]["seed"] == 999

    assert len(data["results"]) == 2

    # Test reproducibility of the whole CLI
    out_file2 = tmp_path / "result2.json"
    test_args[-1] = str(out_file2)
    monkeypatch.setattr("sys.argv", test_args)
    main()

    with open(out_file2, "r") as f:
        data2 = json.load(f)

    assert data == data2


def test_the_script_runs_as_a_script():
    """Run it the way a user does: as a subprocess, not by importing main().

    The in-process CLI test above cannot catch an import error, because pytest
    has already put the repository root on sys.path. The shipped script used
    `from src.exp1...`, which resolves under pytest and raises
    ModuleNotFoundError when the file is executed directly -- so the entry point
    was broken while its own test passed.
    """
    import subprocess
    import tempfile

    root = Path(__file__).parent.parent
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "floor.json"
        proc = subprocess.run(
            [sys.executable, str(root / "scripts" / "audit_word_conditioned_floor.py"),
             "--depths", "2", "--words", "4", "--memories", "4",
             "--bootstrap", "10", "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, f"script failed:\n{proc.stderr[-2000:]}"
        assert out.exists()
        assert "results" in json.loads(out.read_text())
