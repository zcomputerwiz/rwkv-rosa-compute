import os
import sys


def ensure_src_in_path() -> None:
    """Ensures src directory is in sys.path for direct script execution without installation."""
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
