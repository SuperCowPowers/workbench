"""Shared fixtures for the training tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ray_cluster():
    """Let Ray workers import this directory's trial modules.

    The harness owns a Ray session per run and tears it down on the way out, so a fixture
    cannot hold one open for the tests. Workers inherit the driver's environment, so putting
    this directory on ``PYTHONPATH`` here reaches them whichever session they belong to —
    without it every trial dies with ``ModuleNotFoundError`` before the harness is exercised
    at all. ``RAY_num_cpus`` keeps the toy trials from fanning out across every core.
    """
    pytest.importorskip("ray")

    here = str(Path(__file__).parent)
    previous = {key: os.environ.get(key) for key in ("PYTHONPATH", "RAY_num_cpus")}
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [here, previous["PYTHONPATH"]]))
    os.environ["RAY_num_cpus"] = "2"
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
