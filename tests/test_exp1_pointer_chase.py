"""Invariant and leakage tests for the query-deferred pointer chase.

The previous H2 generator shipped without these and three separate defects
reached a reviewer: N changed the input width, the chain executed during
ingestion, and the query sat after the silent positions. Each is checked here.
"""

import itertools
import math
import random
import statistics
from collections import Counter
from fractions import Fraction

import pytest
import torch

from exp1.pointer_chase import (
    ChaseSpec,
    encode_batch,
    execute,
    generate_dataset,
    generate_instance,
    generate_memory,
    make_neutral_vector,
)

M, K = 16, 4


def test_input_width_constant_across_n():
    """Blocker 1: N must not change the architecture."""
    spec = ChaseSpec(num_nodes=M, num_maps=K)
    data = generate_dataset(2, 4, depth=8, seed=0, num_nodes=M, num_maps=K)
    widths = set()
    for n in (0, 1, 8, 32):
        kind = None if n == 0 else "scratchpad"
        x, _ = encode_batch(data, spec, num_silent=n, silent_kind=kind)
        widths.add(x.shape[-1])
        assert x.shape[1] == spec.seq_len(n)
    assert widths == {spec.d_input}, f"input width moved with N: {widths}"


def test_input_width_constant_across_depth():
    """D must not change the architecture either."""
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=32)
    widths = set()
    for d in (1, 4, 16, 32):
        data = generate_dataset(1, 2, depth=d, seed=d, num_nodes=M, num_maps=K)
        x, _ = encode_batch(data, spec)
        widths.add(x.shape[-1])
    assert widths == {spec.d_input}


def test_answer_is_a_function_of_memory_start_and_selectors():
    for inst in generate_dataset(6, 6, depth=8, seed=1, num_nodes=M, num_maps=K):
        assert execute(inst.memory, inst.start, inst.selectors) == inst.answer


def test_memory_alone_does_not_determine_the_answer():
    """Blocker 2: the query must carry the information, not the ingested prefix."""
    rng = random.Random(5)
    mem = generate_memory(rng, num_nodes=M, num_maps=K)
    answers = Counter(generate_instance(rng, mem, depth=8).answer for _ in range(2000))
    assert len(answers) > 1
    # With permutations the answer is uniform given a uniform start.
    assert max(answers.values()) / 2000 < 0.12, "answer concentrates on one node"


def test_permutations_keep_the_floor_at_one_over_m():
    """Arbitrary maps concentrate under iteration; permutations do not."""
    def floor_estimate(permutations, depth, n=4000):
        rng = random.Random(11)
        best = []
        for _ in range(12):
            mem = generate_memory(rng, num_nodes=M, num_maps=K,
                                  permutations=permutations)
            c = Counter(generate_instance(rng, mem, depth=depth).answer
                        for _ in range(n))
            best.append(max(c.values()) / n)
        return statistics.mean(best)

    perm_floor = floor_estimate(True, 8)
    map_floor = floor_estimate(False, 8)
    assert perm_floor < 0.09, f"permutation floor too high: {perm_floor}"
    assert map_floor > perm_floor, "arbitrary maps should concentrate more"


def test_selector_order_matters():
    """If order did not matter the task would not be sequential."""
    rng = random.Random(3)
    mem = generate_memory(rng, num_nodes=M, num_maps=K)
    differing = 0
    for _ in range(200):
        inst = generate_instance(rng, mem, depth=8)
        shuffled = list(inst.selectors)
        rng.shuffle(shuffled)
        if execute(mem, inst.start, shuffled) != inst.answer:
            differing += 1
    assert differing > 100, "permuting the selectors rarely changes the answer"


def test_arms_differ_only_at_silent_rows():
    spec = ChaseSpec(num_nodes=M, num_maps=K)
    data = generate_dataset(2, 3, depth=4, seed=2, num_nodes=M, num_maps=K)
    nv = make_neutral_vector(spec)
    n = 7
    xb, ab = encode_batch(data, spec, num_silent=n, silent_kind="scratchpad")
    xc, ac = encode_batch(data, spec, num_silent=n, silent_kind="neutral",
                          neutral_vector=nv)
    assert torch.equal(ab, ac)
    differing_rows = (xb != xc).any(dim=-1)
    assert int(differing_rows.sum()) == len(data) * n
    first_silent = spec.num_nodes + 1
    assert bool(differing_rows[:, first_silent:first_silent + n].all())


def test_neutral_vector_is_norm_matched_and_fixed():
    spec = ChaseSpec(num_nodes=M, num_maps=K)
    a = make_neutral_vector(spec, seed=0)
    b = make_neutral_vector(spec, seed=0)
    assert torch.equal(a, b), "neutral vector is not deterministic"
    scratch = torch.zeros(spec.d_input)
    scratch[spec.off_kind + 2] = 1.0
    assert abs(float(a.norm() - scratch.norm())) < 1e-5


def test_zero_arm_carries_no_input_signal():
    spec = ChaseSpec(num_nodes=M, num_maps=K)
    data = generate_dataset(1, 2, depth=4, seed=4, num_nodes=M, num_maps=K)
    x, _ = encode_batch(data, spec, num_silent=5, silent_kind="zero")
    first_silent = spec.num_nodes + 1
    assert float(x[:, first_silent:first_silent + 5].abs().sum()) == 0.0


def test_query_precedes_the_silent_positions():
    """Blocker 3: silent tokens are useless if the question comes after them."""
    spec = ChaseSpec(num_nodes=M, num_maps=K)
    data = generate_dataset(1, 1, depth=4, seed=6, num_nodes=M, num_maps=K)
    x, _ = encode_batch(data, spec, num_silent=3, silent_kind="scratchpad")
    kinds = x[0, :, spec.off_kind:spec.off_kind + 6].argmax(dim=-1).tolist()
    query_at = kinds.index(1)
    silent_at = [i for i, k in enumerate(kinds) if k == 2]
    probe_at = kinds.index(5)
    assert query_at < min(silent_at) < max(silent_at) < probe_at


def test_num_silent_requires_a_kind():
    spec = ChaseSpec(num_nodes=M, num_maps=K)
    data = generate_dataset(1, 1, depth=2, seed=7, num_nodes=M, num_maps=K)
    with pytest.raises(ValueError):
        encode_batch(data, spec, num_silent=4)

def test_data_generation_depth_invariance():
    """Defect 2: Changing depth must not change the sampled memories, starts, or selector prefixes."""
    seed = 42
    d2 = generate_dataset(2, 4, depth=2, seed=seed)
    d4 = generate_dataset(2, 4, depth=4, seed=seed)
    d8 = generate_dataset(2, 4, depth=8, seed=seed)

    for i in range(len(d2)):
        assert d2[i].memory == d4[i].memory == d8[i].memory, "Memories differ across depths"
        assert d2[i].start == d4[i].start == d8[i].start, "Starts differ across depths"

        # d2 selectors should be prefix of d4
        assert d4[i].selectors[:2] == d2[i].selectors
        # d4 selectors should be prefix of d8
        assert d8[i].selectors[:4] == d4[i].selectors


# --- the query-aware floor -------------------------------------------------
#
# These are exhaustive over a small symmetric group rather than sampled. The
# claims are algebraic, so a Monte Carlo test would only add flakiness, and an
# earlier version of this file asserted a scientific null as a CI pass
# condition, which is not something a test suite should be deciding.

def _compose(outer, inner):
    """outer after inner, both as images of 0..M-1."""
    return tuple(outer[x] for x in inner)


def _power(perm, exponent):
    out = tuple(range(len(perm)))
    for _ in range(exponent):
        out = _compose(perm, out)
    return out


def test_a_selector_used_exactly_once_is_sufficient_for_uniformity():
    """Exact, by exhaustion: pi -> A pi B is a bijection of S_M.

    If one selector appears exactly once, conditioning on every other map
    leaves the word in the form A pi_j B with A and B fixed. Composing a
    uniform group element with fixed elements is uniform, so the answer
    distribution is exactly 1/M. No commutativity is required, and repeats
    among the *other* selectors are irrelevant.
    """
    m = 4
    perms = list(itertools.permutations(range(m)))
    for a in perms[::7]:                       # a few fixed prefixes
        for b in perms[::11]:                  # a few fixed suffixes
            reached = Counter(_compose(b, _compose(free, a)) for free in perms)
            assert set(reached) == set(perms), "not onto S_M"
            assert set(reached.values()) == {1}, "not uniform over S_M"


def test_the_singleton_condition_is_sufficient_but_not_necessary():
    """The all-identical word pi^D is uniform whenever gcd(D, exponent) = 1.

    A previous version of this file claimed uniformity holds *iff* some
    selector appears exactly once. It does not. In S_4 the exponent is
    lcm(1,2,3,4) = 12, so raising to any power coprime to 12 is a bijection,
    and pi^5 is exactly uniform despite having no singleton selector.
    """
    m = 4
    perms = list(itertools.permutations(range(m)))
    exponent = 12
    assert math.gcd(5, exponent) == 1
    reached = Counter(_power(perm, 5) for perm in perms)
    assert set(reached) == set(perms)
    assert set(reached.values()) == {1}, "pi^5 is not uniform on S_4"

    # And a power that is *not* coprime concentrates, which is the case the
    # floor is about.
    assert math.gcd(2, exponent) != 1
    squares = Counter(_power(perm, 2) for perm in perms)
    assert len(squares) < len(perms), "squaring should not be onto"


def test_expected_fixed_points_counts_only_divisors_at_most_M():
    """E[fix(pi^D)] = #{l <= M : l | D}, exactly, over all of S_M.

    Divisors larger than M cannot be cycle lengths, so the unrestricted
    divisor function d(D) overcounts as soon as D exceeds M. A previous
    version of this file used d(D), which is right only for D <= M.
    """
    for m in (4, 5, 6):
        perms = list(itertools.permutations(range(m)))
        for depth in range(1, 13):
            total = sum(sum(1 for i, x in enumerate(_power(perm, depth)) if x == i)
                        for perm in perms)
            measured = Fraction(total, len(perms))
            restricted = sum(1 for length in range(1, m + 1) if depth % length == 0)
            assert measured == restricted, (
                f"M={m} D={depth}: expected {restricted}, got {measured}")

            unrestricted = sum(1 for length in range(1, depth + 1) if depth % length == 0)
            if unrestricted != restricted:
                assert measured != unrestricted, (
                    "the unrestricted divisor count should be wrong here")
