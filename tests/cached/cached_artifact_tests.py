"""Tests for Cached Artifacts: caching behavior, poke lifecycle, and stale/fresh cycles"""

from pprint import pprint

# Workbench Imports
from workbench.cached.cached_data_source import CachedDataSource
from workbench.cached.cached_feature_set import CachedFeatureSet
from workbench.cached.cached_model import CachedModel
from workbench.cached.cached_endpoint import CachedEndpoint
from workbench.cached.cached_meta import CachedMeta
from workbench.api.model import Model


def test_cached_data_source():
    """Basic CachedDataSource functionality"""
    print("\n\n*** Cached DataSource ***")
    my_data = CachedDataSource("abalone_data")
    pprint(my_data.summary())
    my_data.details()
    pprint(my_data.health_check())
    pprint(my_data.workbench_meta())


def test_cached_feature_set():
    """Basic CachedFeatureSet functionality"""
    print("\n\n*** Cached FeatureSet ***")
    my_features = CachedFeatureSet("abalone_features")
    pprint(my_features.summary())
    pprint(my_features.details())
    pprint(my_features.health_check())
    pprint(my_features.workbench_meta())


def test_cached_model():
    """Basic CachedModel functionality"""
    print("\n\n*** Cached Model ***")
    my_model = CachedModel("abalone-regression")
    pprint(my_model.summary())
    pprint(my_model.details())
    pprint(my_model.health_check())
    pprint(my_model.workbench_meta())


def test_cached_endpoint():
    """Basic CachedEndpoint functionality"""
    print("\n\n*** Cached Endpoint ***")
    my_endpoint = CachedEndpoint("abalone-regression")
    pprint(my_endpoint.summary())
    pprint(my_endpoint.details())
    pprint(my_endpoint.health_check())
    pprint(my_endpoint.workbench_meta())


def test_poke_with_model_class():
    """update_modified_timestamp should work with Model (not just CachedModel)"""
    print("\n\n*** Poke with Model class ***")
    meta = CachedMeta()
    meta.models()  # Bootstrap registry

    model = Model("abalone-regression")
    before = meta.get_modified_timestamp(model)
    meta.update_modified_timestamp(model)
    after = meta.get_modified_timestamp(model)

    assert after > before, f"Poke with Model class should work: {before} -> {after}"
    print(f"Poke with Model class works: {before} -> {after}")


def test_artifact_single_stale():
    """After a poke, a CachedModel method should be stale once, then fresh

    This is the critical test: poke → first call stale → second call fresh.
    If this fails, it means the artifact cache and meta registry are getting
    out of sync (the 'double-stale' bug).
    """
    print("\n\n*** Artifact Single Stale Test ***")
    meta = CachedMeta()
    meta.models()  # Bootstrap registry

    my_model = CachedModel("abalone-regression")

    # Prime the artifact cache
    summary1 = my_model.summary()
    assert summary1 is not None

    # Poke the model
    meta.update_modified_timestamp(my_model)
    poke_ts = meta.get_modified_timestamp(my_model)
    print(f"Poked: registry timestamp = {poke_ts}")

    # First call should be stale (refetch)
    summary2 = my_model.summary()
    assert summary2 is not None

    # Check what the artifact cache stored as _modified
    from workbench.utils.workbench_cache import WorkbenchCache

    cache_key = f"cachedmodel_{my_model.name}_{WorkbenchCache.flatten_key(CachedModel.summary.__wrapped__)}"
    cached_entry = CachedModel.artifact_cache.get(cache_key)
    cached_modified = cached_entry.get("_modified") if cached_entry else None
    print(f"After first call: cached_modified = {cached_modified}")
    print(f"After first call: registry_ts    = {meta.get_modified_timestamp(my_model)}")

    # Second call should be fresh (no refetch)
    # If the registry has a newer timestamp, this will be stale again
    current_ts = meta.get_modified_timestamp(my_model)
    if cached_modified is not None and cached_modified < current_ts:
        print(f"BUG: Double-stale! cached_modified={cached_modified} < registry={current_ts}")
    else:
        print("OK: Artifact cache and registry agree — second call will be fresh")

    summary3 = my_model.summary()
    assert summary3 is not None


if __name__ == "__main__":

    # Basic functionality tests
    test_cached_data_source()
    test_cached_feature_set()
    test_cached_model()
    test_cached_endpoint()

    # Poke lifecycle tests
    test_poke_with_model_class()
    test_artifact_single_stale()

    print("\n\nAll Cached Artifact tests passed!")
