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
    """challenger_models() lists the model inputs of the endpoint's promotion node"""
    meta = CachedMeta()
    assert meta.challenger_models("aqsol-regression") == ["aqsol-regression-1", "aqsol-regression-2"]
    assert meta.challenger_models("aqsol-class") == ["aqsol-class-1", "aqsol-class-2"]

    # Endpoints without a promotion node have no challengers
    assert meta.challenger_models("abalone-regression") == []


if __name__ == "__main__":
    test_champion_models()
    test_challenger_models()
    print("All CachedMeta API tests passed!")
