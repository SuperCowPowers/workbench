"""Tests for the CachedMeta API methods (as opposed to the caching mechanics,
which live in cached_meta_tests.py). Tests always use CachedMeta, not Meta."""

# Workbench Imports
from workbench.cached.cached_meta import CachedMeta


def test_champion_models():
    """champion_models() returns a [Model, Endpoint] row for each promotion endpoint"""
    meta = CachedMeta()
    champs = meta.champion_models()
    assert list(champs.columns) == ["Model", "Endpoint"]

    # The aqsol pipelines have model_promotion nodes, so their endpoints are champions
    assert {"aqsol-regression", "aqsol-class"} <= set(champs["Endpoint"])


def test_challenger_models():
    """challenger_models() lists the model inputs of the endpoint's promotion node

    A contest accumulates entrants over time, so this asserts the seeded challengers are
    present rather than pinning an exact list — a new entrant is expected, not a failure.
    """
    meta = CachedMeta()
    assert {"aqsol-regression-1", "aqsol-regression-2"} <= set(meta.challenger_models("aqsol-regression"))
    assert {"aqsol-class-1", "aqsol-class-2"} <= set(meta.challenger_models("aqsol-class"))

    # Endpoints without a promotion node have no challengers
    assert meta.challenger_models("abalone-regression") == []


if __name__ == "__main__":
    test_champion_models()
    test_challenger_models()
    print("All CachedMeta API tests passed!")
