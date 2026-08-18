from importlib.metadata import PackageNotFoundError, version

from .blinkdl_reference import blinkdl_rosa_4bit_reference, rosa_slow_ref
from .checkpoint import (
    inspect_checkpoint,
    load_rosa_checkpoint,
    validate_checkpoint_state_dict,
)
from .config import DEFAULT_CONFIG, ROSAConfig
from .diagnostics import get_environment_info, print_diagnostics
from .model import ROSAModelSkeleton
from .rosa_compat import ROSALayerCompat, apply_blinkdl_embedding, rosa_4bit_forward

try:
    __version__ = version("rosa-compute")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "ROSAConfig",
    "DEFAULT_CONFIG",
    "ROSAModelSkeleton",
    "blinkdl_rosa_4bit_reference",
    "rosa_slow_ref",
    "rosa_4bit_forward",
    "apply_blinkdl_embedding",
    "ROSALayerCompat",
    "inspect_checkpoint",
    "load_rosa_checkpoint",
    "validate_checkpoint_state_dict",
    "get_environment_info",
    "print_diagnostics",
]
