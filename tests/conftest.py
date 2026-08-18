import pytest


@pytest.fixture
def sample_config():
    from rosa_compute import ROSAConfig
    return ROSAConfig()


@pytest.fixture
def exact_match_fixture():
    """Exact Match Fixture:
    Query suffix occurs exactly once in key.
    Q symbols: [1, 2, 3, 4]
    K symbols: [1, 2, 3, 5]
    V symbols: [10, 20, 30, 40]

    At t=2 (Q symbol 3, suffix [1, 2, 3]):
    Matches K[0:3] ([1, 2, 3]). Match length = 3. Target route s = 3 -> V[3] symbol 40.
    Values returned:
    t=0: no prev matches -> unmatched (0)
    t=1: Q suffix [1, 2] matches K[0:2] -> s=2 -> V[2]=30
    t=2: Q suffix [1, 2, 3] matches K[0:3] -> s=3 -> V[3]=40
    t=3: Q suffix [4] -> no match -> 0
    """
    return {
        "q_sym": [1, 2, 3, 4],
        "k_sym": [1, 2, 3, 5],
        "v_sym": [10, 20, 30, 40],
        "expected_idx": [0, 30, 40, 0],
        "expected_ln": [0, 2, 3, 0],
    }


@pytest.fixture
def latest_match_fixture():
    """Latest Match Fixture (Recency Tie-Breaking):
    Same suffix occurs at multiple valid locations.
    Q symbols: [5, 5, 0]
    K symbols: [5, 5, 0]
    V symbols: [10, 20, 30]

    At t=1 (Q symbol 5, suffix [5]):
    Matches K[0] (s=1, V[1]=20) and K[1] (s=2, V[2]=30).
    Latest valid occurrence wins -> route s=2 -> V[2]=30, match length = 1.
    """
    return {
        "q_sym": [5, 5, 0],
        "k_sym": [5, 5, 0],
        "v_sym": [10, 20, 30],
        "expected_idx": [0, 20, 0],
        "expected_ln": [0, 1, 0],
    }


@pytest.fixture
def longest_match_fixture():
    """Longest Match Precedence Fixture:
    Shorter suffix occurs at a later location while a longer suffix occurs earlier.
    Q symbols: [1, 2, 1, 2]
    K symbols: [1, 2, 1, 2]
    V symbols: [10, 20, 30, 40]

    At t=2: Q suffix [1, 2, 1], matches K[0:1]=[1] -> len=1, s=1 -> V[1]=20.
    At t=3: Q suffix [1, 2, 1, 2]:
        w=2: suffix [1, 2], matches K[0:2] = [1, 2]. Route s = 0+2 = 2, V[2] = 30, len = 2.
        w=1: suffix [2], matches K[1] = [2]. Route s = 1+1 = 2, V[2] = 30, len = 1.
        Longest match (len=2) takes precedence over shorter match (len=1).
    """
    return {
        "q_sym": [1, 2, 1, 2],
        "k_sym": [1, 2, 1, 2],
        "v_sym": [10, 20, 30, 40],
        "expected_idx": [0, 0, 20, 30],
        "expected_ln": [0, 0, 1, 2],
    }
