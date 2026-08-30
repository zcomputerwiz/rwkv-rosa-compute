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
    # An all-identical word at depth where composition concentrates.
    # Say depth 32, word is just '0' repeated 32 times.
    # Iterating the *same* random permutation 32 times can have varying effects,
    # but wait: iterating a permutation does not concentrate onto attractors!
    # A permutation is bijective.
    # Let's think: The problem states "arbitrary random functions concentrate... permutations don't."
    # Wait, the prompt says:
    # "(b) an all-identical word at a depth where the composition concentrates yields a B_w clearly above 1/M"
    # Wait, permutations DO NOT concentrate under iteration. BUT the prompt says "an all-identical word at a depth where the composition concentrates".
    # Wait, does the generator use arbitrary maps or permutations?
    # generate_memory(..., permutations=True) is used in pointer_chase.py, but maybe it doesn't have to be True for the test, or wait!
    # Ah. The prompt says: "an all-identical word at a depth where the composition concentrates yields a B_w clearly above 1/M".
    # Wait. If permutations are used, does it concentrate?
    # No, permutations do not concentrate.
    # What DOES concentrate? P(answer == s | w, s) for an all-identical word!
    # Wait. If w = (0, 0, ..., 0) repeated D times, what is P(answer == s | w, s)?
    # For a given memory, the map 0 is a permutation.
    # The order of a permutation on 16 elements divides 16!
    # But for a specific depth D, say D=16, the orbits of size dividing 16 will return to the start node!
    # So for D=16, any orbit of length 1, 2, 4, 8, 16 returns to itself after 16 steps.
    # So P(answer == s | w, s) will be unusually high! It's exactly the fixed-point fraction.

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

    # Because of orbits dividing 16, P(answer == s) should be > 1/M.
    assert e_bw > 1.0 / M + 0.05

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
