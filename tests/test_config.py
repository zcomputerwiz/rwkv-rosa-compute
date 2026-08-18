import pytest
import torch

from rosa_compute import DEFAULT_CONFIG, ROSAConfig


def test_default_config():
    assert DEFAULT_CONFIG.n_layer == 12
    assert DEFAULT_CONFIG.n_embd == 768
    assert DEFAULT_CONFIG.vocab_size == 65536
    assert DEFAULT_CONFIG.rosa_bits == 4
    assert DEFAULT_CONFIG.rosa_groups == 192
    assert DEFAULT_CONFIG.context_length == 512
    assert DEFAULT_CONFIG.dtype == torch.float16


def test_config_validation():
    # Invalid n_layer
    with pytest.raises(ValueError, match="n_layer must be >= 1"):
        ROSAConfig(n_layer=0)

    # Invalid n_embd
    with pytest.raises(ValueError, match="n_embd must be >= 4"):
        ROSAConfig(n_embd=2)

    # Invalid vocab_size
    with pytest.raises(ValueError, match="vocab_size must be >= 1"):
        ROSAConfig(vocab_size=0)

    # Invalid rosa_bits
    with pytest.raises(ValueError, match="rosa_bits must be 4"):
        ROSAConfig(rosa_bits=2)

    # Indivisible n_embd
    with pytest.raises(ValueError, match="must be divisible by rosa_bits"):
        ROSAConfig(n_embd=769, rosa_groups=192)

    # Mismatched groups
    with pytest.raises(ValueError, match="rosa_groups .* must equal n_embd / rosa_bits"):
        ROSAConfig(rosa_groups=100)

    # Invalid context_length
    with pytest.raises(ValueError, match="context_length must be >= 1"):
        ROSAConfig(context_length=0)

    # Invalid dtype
    with pytest.raises(ValueError, match="dtype must be a floating-point PyTorch dtype"):
        ROSAConfig(dtype=torch.int32)
