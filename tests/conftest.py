from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    """Keep all test_exp0_* modules in the Experiment 0 CI partition."""
    for item in items:
        if Path(str(item.fspath)).name.startswith("test_exp0_"):
            item.add_marker(pytest.mark.exp0)


@pytest.fixture
def sample_config():
    from rosa_compute import ROSAConfig

    return ROSAConfig()


@pytest.fixture
def exact_match_fixture():
    """Exact Match Fixture:
    Query suffix occurs at a single unique position.
    Q symbols: [0, 1, 2]
    K symbols: [1, 2, 0, 0]
    V symbols: [10, 20, 30, 40]

    Step-by-step trace for rosa_slow_ref(q, k, v):
    i=0 (q=[0]): w=1 -> t=[0]. j in [0]: k[0]=1 != 0. No match -> idx[0]=0, ln[0]=0.
    i=1 (q=[0, 1]):
      w=2: t=[0, 1]. j in []: no loop.
      w=1: t=[1]. j=0 -> k[0]=1 == 1 -> MATCH!
           j=0, w=1 -> s = j + w = 0 + 1 = 1 -> v[1] = 20, ln[1] = 1.
    i=2 (q=[0, 1, 2]):
      w=3: t=[0, 1, 2]. j in []: no loop.
      w=2: t=[1, 2]. j=0 -> k[0:2]=[1, 2] == [1, 2] -> MATCH!
           j=0, w=2 -> s = j + w = 0 + 2 = 2 -> v[2] = 30, ln[2] = 2.

    Expected idx: [0, 20, 30]
    Expected ln:  [0, 1, 2]
    """
    return {
        "q_sym": [0, 1, 2],
        "k_sym": [1, 2, 0, 0],
        "v_sym": [10, 20, 30, 40],
        "expected_idx": [0, 20, 30],
        "expected_ln": [0, 1, 2],
    }


@pytest.fixture
def latest_match_fixture():
    """Latest Match Fixture (Equal-length Recency Tie-Breaking):
    Two distinct prior starting positions j1 < j2 <= i-w produce the same maximum suffix length w.
    Q symbols: [0, 0, 7]
    K symbols: [7, 7, 0, 0]
    V symbols: [10, 20, 30, 40]

    Step-by-step trace at i=2 (q=[0, 0, 7]):
    w=1: t=[7]. Inner loop j ranges from i-w = 2-1 = 1 down to 0:
      j=1: k[1:2] = [7] == [7] -> MATCH!
           Since inner loop searches j in reverse order (1 down to 0), j=1 (latest) is found FIRST!
           j=1, w=1 -> route s = j + w = 1 + 1 = 2 -> v[2] = 30, ln[2] = 1.
      (j=0 would give s = 0 + 1 = 1 -> v[1] = 20, but j=1 wins because it is the latest valid position).

    Expected idx: [0, 0, 30]
    Expected ln:  [0, 0, 1]
    """
    return {
        "q_sym": [0, 0, 7],
        "k_sym": [7, 7, 0, 0],
        "v_sym": [10, 20, 30, 40],
        "expected_idx": [0, 0, 30],
        "expected_ln": [0, 0, 1],
    }


@pytest.fixture
def longest_match_fixture():
    """Longest Match Precedence Fixture:
    A shorter suffix match exists at a later position (j=1, w=1)
    while a longer suffix match exists at an earlier position (j=0, w=2).
    Q symbols: [1, 2, 1, 2]
    K symbols: [1, 2, 0, 2, 0]
    V symbols: [10, 20, 30, 40, 50]

    Step-by-step trace at i=3 (q=[1, 2, 1, 2]):
    w=4: t=[1, 2, 1, 2]. j in []: no loop.
    w=3: t=[2, 1, 2]. j=0 -> k[0:3] = [1, 2, 0] != [2, 1, 2].
    w=2: t=[1, 2]. Inner loop j ranges from 3-2 = 1 down to 0:
      j=1: k[1:3] = [2, 0] != [1, 2].
      j=0: k[0:2] = [1, 2] == [1, 2] -> MATCH!
           s = j + w = 0 + 2 = 2 -> v[2] = 30, ln[3] = 2.
    w=1: (Tested only if w=2 failed, but w=2 already matched!).
         Note: at w=1, t=[2], j=1 would match k[1]=[2] (s=2 -> v[2]=30, len=1).
         The longer match (w=2 at j=0) wins over the shorter match (w=1 at j=1).

    Expected idx: [0, 0, 20, 30]
    Expected ln:  [0, 0, 1, 2]
    """
    return {
        "q_sym": [1, 2, 1, 2],
        "k_sym": [1, 2, 0, 2, 0],
        "v_sym": [10, 20, 30, 40, 50],
        "expected_idx": [0, 0, 20, 30],
        "expected_ln": [0, 0, 1, 2],
    }
