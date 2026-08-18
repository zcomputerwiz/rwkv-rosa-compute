#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rosa_compute import print_diagnostics

if __name__ == "__main__":
    print_diagnostics()
