"""Helpers for verifying that pipeline_utils reaches a pipeline run."""

import os
import sys


def describe_environment() -> dict:
    """Where this package was imported from, and how it got on the path.

    Returns:
        dict: Import origin, the staging ref (Batch only), PYTHONPATH, interpreter.
    """
    return {
        "import_origin": os.path.dirname(os.path.abspath(__file__)),
        "utils_ref": os.environ.get("ML_PIPELINE_UTILS", "<unset: local run>"),
        "pythonpath": os.environ.get("PYTHONPATH", "<unset>"),
        "python": sys.executable,
    }


def add(a: int, b: int) -> int:
    """Trivial callable proving the module executes, not just imports."""
    return a + b
