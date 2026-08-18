def test_imports():
    import rosa_compute
    assert hasattr(rosa_compute, "ROSAConfig")
    assert hasattr(rosa_compute, "DEFAULT_CONFIG")
    assert hasattr(rosa_compute, "blinkdl_rosa_4bit_reference")
    assert hasattr(rosa_compute, "rosa_slow_ref")
    assert hasattr(rosa_compute, "rosa_4bit_forward")
    assert hasattr(rosa_compute, "apply_blinkdl_embedding")
    assert hasattr(rosa_compute, "ROSALayerCompat")
    assert hasattr(rosa_compute, "load_rosa_checkpoint")
    assert hasattr(rosa_compute, "validate_checkpoint_state_dict")
    assert hasattr(rosa_compute, "get_environment_info")
    assert hasattr(rosa_compute, "print_diagnostics")
    assert rosa_compute.__version__ == "0.1.0"
