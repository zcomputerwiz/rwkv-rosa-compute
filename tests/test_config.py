import pytest

from rosa_compute import DEFAULT_CONFIG, ROSAConfig


def test_default_config():
    assert DEFAULT_CONFIG.n_layer == 12
    assert DEFAULT_CONFIG.n_embd == 768
    assert DEFAULT_CONFIG.vocab_size == 65536
    assert DEFAULT_CONFIG.rosa_bits == 4
    assert DEFAULT_CONFIG.rosa_groups == 192
    assert DEFAULT_CONFIG.context_length == 512

def test_config_validation():
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
