"""Shared fixtures for the training tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ray_cluster():
    """A Ray cluster whose workers can import this directory's trial modules.

    ``_run_ray`` calls ``ray.init(..., ignore_reinit_error=True)``, so initializing first
    here wins: the runtime env set below is the one the workers get. Adding this directory
    to their ``PYTHONPATH`` is what lets a worker import ``hpo_ray_trials`` — without it
    every trial dies with ``ModuleNotFoundError`` before the harness is exercised at all.
    """
    ray = pytest.importorskip("ray")

    ray.init(
        runtime_env={"env_vars": {"PYTHONPATH": str(Path(__file__).parent)}},
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        configure_logging=False,
    )
    yield
    ray.shutdown()
