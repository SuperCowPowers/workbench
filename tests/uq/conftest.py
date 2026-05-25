"""Pytest fixtures wrapping the regime registry in :mod:`tests.uq.fixtures`.

A test that wants the ``well_calibrated`` regime can ask for it directly::

    def test_v0_coverage(uq_well_calibrated):
        uq = UQModelV0.fit(
            uq_well_calibrated.y_true_val,
            uq_well_calibrated.y_pred_val,
            uq_well_calibrated.prediction_std_val,
        )
        ...

Each named-regime fixture is built fresh per test with seed=0 — so the same
fixture name always yields the same data. If a test needs a different seed
(e.g. for parameterized robustness sweeps), use :func:`uq_fixture_factory`
instead.
"""

from __future__ import annotations

from typing import Callable

import pytest

from .fixtures import UQFixture


# ---------------------------------------------------------------------------
# Factory fixture — for parametric tests that want a different seed/regime
# ---------------------------------------------------------------------------
@pytest.fixture
def uq_fixture_factory() -> Callable[..., UQFixture]:
    """Return a builder ``(regime, seed) -> UQFixture``.

    Use when a test wants to sweep across regimes or seeds rather than wire
    up nine separate fixtures.
    """
    return UQFixture.make


# ---------------------------------------------------------------------------
# Named-regime fixtures (seed=0)
# ---------------------------------------------------------------------------
@pytest.fixture
def uq_well_calibrated() -> UQFixture:
    return UQFixture.make("well_calibrated", seed=0)


@pytest.fixture
def uq_overconfident() -> UQFixture:
    return UQFixture.make("overconfident", seed=0)


@pytest.fixture
def uq_underconfident() -> UQFixture:
    return UQFixture.make("underconfident", seed=0)


@pytest.fixture
def uq_heteroskedastic() -> UQFixture:
    return UQFixture.make("heteroskedastic", seed=0)


@pytest.fixture
def uq_biased() -> UQFixture:
    return UQFixture.make("biased", seed=0)


@pytest.fixture
def uq_ood_queries() -> UQFixture:
    return UQFixture.make("out_of_distribution_queries", seed=0)


@pytest.fixture
def uq_activity_cliff_queries() -> UQFixture:
    return UQFixture.make("activity_cliff_queries", seed=0)


@pytest.fixture
def uq_bin_edge_queries() -> UQFixture:
    return UQFixture.make("bin_edge_queries", seed=0)


@pytest.fixture
def uq_tiny_calibration_set() -> UQFixture:
    return UQFixture.make("tiny_calibration_set", seed=0)
