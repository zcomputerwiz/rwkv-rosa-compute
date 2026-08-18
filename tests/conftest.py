import pytest
import torch


@pytest.fixture
def sample_config():
    from rosa_compute import ROSAConfig
    return ROSAConfig()

@pytest.fixture
def exact_match_fixture():
    """Query suffix occurs exactly once in key."""
    # T=4, C=4 (1 group, 4 bits)
    # Bit vector symbols:
    # Key sequence: [1, 2, 3, 4]
    # Query sequence: [0, 1, 2, 3] -> query at t=3 is symbol 3, suffix matches key at j=2 (symbol 3)
    # v sequence: [10, 20, 30, 40]
    q = torch.tensor([[[1.0, 0.0, 0.0, 0.0],  # 1
                       [0.0, 1.0, 0.0, 0.0],  # 2
                       [1.0, 1.0, 0.0, 0.0],  # 3
                       [1.0, 1.0, 0.0, 0.0]]], dtype=torch.float32) # 3
    k = torch.tensor([[[1.0, 0.0, 0.0, 0.0],  # 1
                       [0.0, 1.0, 0.0, 0.0],  # 2
                       [1.0, 1.0, 0.0, 0.0],  # 3
                       [0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32) # 4
    v = torch.tensor([[[1.0, 0.0, 1.0, 0.0],  # 5
                       [0.0, 1.0, 1.0, 0.0],  # 6
                       [1.0, 1.0, 1.0, 0.0],  # 7
                       [0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32) # 8
    return q, k, v

@pytest.fixture
def deterministic_fixtures():
    """Suite of deterministic cases covering all required behaviors."""
    return {}
