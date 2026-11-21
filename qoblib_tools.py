"""Compatibility import for notebooks run directly from the SharQ checkout."""
import os
from pathlib import Path
import sys

from packages.qoblib_tools import qoblib_tools as _tools

if not os.environ.get("QOBLIB_DATA_DIR"):
    _tools.DATA_DIR = Path(__file__).resolve().parent / "qoblib_data"
sys.modules[__name__] = _tools
