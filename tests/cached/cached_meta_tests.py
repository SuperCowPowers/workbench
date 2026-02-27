"""Tests for the CachedMeta caching system: registry, incremental refresh, and poke lifecycle"""

import time
from datetime import datetime, timezone
from pprint import pprint

# Workbench Imports
from workbench.cached.cached_meta import CachedMeta
from workbench.cached.cached_model import CachedModel
from workbench.cached.cached_endpoint import CachedEndpoint
from workbench.api.model import Model
from workbench.api.endpoint import Endpoint


def test_registry_bootstrap():
    """Calling a list method should populate the modified registry"""
    meta = CachedMeta()

    # Call models (details=False) to bootstrap the registry
    df = meta.models()
    assert not df.empty, "models() should return a non-empty DataFrame"

    # Registry should now have entries for models
    registry = meta.get_modified_registry("models")
    assert len(registry) > 0, "Registry should have model entries after calling models()"
    print(f"Registry bootstrapped with {len(registry)} models")

    # Each entry should be a datetime
    for name, ts in list(registry.items())[:3]:
        assert isinstance(ts, datetime), f"Registry entry for {name} should be a datetime, got {type(ts)}"
        print(f"  {name}: {ts}")


def test_registry_resolve():
    """_resolve_registry_key should work with Model, CachedModel, Endpoint, etc."""
    meta = CachedMeta()

    # Test with different artifact types
    model = Model("abalone-regression")
    cached_model = CachedModel("abalone-regression")
    endpoint = Endpoint("abalone-regression")
    cached_endpoint = CachedEndpoint("abalone-regression")

    assert meta._resolve_registry_key(model) == "models"
    assert meta._resolve_registry_key(cached_model) == "models"
    assert meta._resolve_registry_key(endpoint) == "endpoints"
    assert meta._resolve_registry_key(cached_endpoint) == "endpoints"
    print("All artifact types resolve correctly")


def test_get_modified_timestamp():
    """get_modified_timestamp should work with any artifact type"""
    meta = CachedMeta()

    # Bootstrap registry
    meta.models()

    # Both Model and CachedModel should return the same timestamp
    model = Model("abalone-regression")
    cached_model = CachedModel("abalone-regression")

    ts_from_model = meta.get_modified_timestamp(model)
    ts_from_cached = meta.get_modified_timestamp(cached_model)

    assert ts_from_model is not None, "Should find timestamp for Model"
    assert ts_from_model == ts_from_cached, "Model and CachedModel should return same timestamp"
    print(f"get_modified_timestamp works for both Model and CachedModel: {ts_from_model}")


def test_poke():
    """update_modified_timestamp should update the registry entry to now"""
    meta = CachedMeta()

    # Bootstrap registry
    meta.models()

    model = Model("abalone-regression")
    before = meta.get_modified_timestamp(model)
    assert before is not None

    # Poke the model
    meta.update_modified_timestamp(model)
    after = meta.get_modified_timestamp(model)

    assert after > before, f"Poke should set timestamp to now: {before} -> {after}"
    print(f"Poke updated registry: {before} -> {after}")


def test_poke_triggers_detail_refresh():
    """After a poke, models(details=True) should refetch the poked model"""
    meta = CachedMeta()

    # Prime the detail cache by calling details=True
    df1 = meta.models(details=True)
    assert not df1.empty

    # Wait for TTL to expire so next call actually runs _refresh_details
    time.sleep(meta._cache_ttl + 1)

    # Poke the model
    model = Model("abalone-regression")
    meta.update_modified_timestamp(model)

    # Call details=True again — should refetch the poked model
    df2 = meta.models(details=True)
    assert not df2.empty

    # The poked model should still be in the result
    name_col = "Model Group"
    assert "abalone-regression" in df2[name_col].values
    print("Poke triggered detail refresh successfully")


def test_poke_single_stale_cycle():
    """After a poke and one refresh, the registry should be stable (no double-stale)

    This tests the interaction between the meta cache registry and the artifact cache.
    After poke → refresh, both caches should agree on timestamps.
    """
    meta = CachedMeta()

    # Prime the detail cache
    meta.models(details=True)
    time.sleep(meta._cache_ttl + 1)

    # Poke
    model = Model("abalone-regression")
    meta.update_modified_timestamp(model)
    poke_ts = meta.get_modified_timestamp(model)

    # First refresh — should detect stale and refetch
    meta.models(details=True)
    time.sleep(meta._cache_ttl + 1)

    # After the refresh, the registry timestamp should be stable
    post_refresh_ts = meta.get_modified_timestamp(model)

    # Second refresh — should NOT detect stale again
    meta.models(details=True)
    post_second_ts = meta.get_modified_timestamp(model)

    # The registry should be stable after the first refresh
    assert post_refresh_ts == post_second_ts, (
        f"Registry should be stable after refresh: {post_refresh_ts} != {post_second_ts}"
    )
    print(f"Registry stable: poke={poke_ts}, after_refresh={post_refresh_ts}")


def test_deleted_artifact():
    """Artifacts deleted from AWS should be removed from cached details"""
    meta = CachedMeta()

    # Get the current model list
    df = meta.models(details=True)
    model_count = len(df)
    print(f"Current model count: {model_count}")

    # The models in the DataFrame should match what's in AWS
    # (we can't easily test deletion without actually deleting, but we can verify
    # that the cached details only contain models from the lightweight list)
    lightweight = meta.models(details=False)
    assert set(df["Model Group"]) == set(lightweight["Model Group"]), (
        "Detail and lightweight lists should have the same model names"
    )
    print("Detail and lightweight lists match")


def test_registry_full_dump():
    """get_modified_registry should return all artifact types"""
    meta = CachedMeta()

    # Bootstrap all registries
    meta.data_sources()
    meta.feature_sets()
    meta.models()
    meta.endpoints()

    registry = meta.get_modified_registry()

    # Should have entries for all four artifact types
    for artifact_type in ["data_sources", "feature_sets", "models", "endpoints"]:
        assert artifact_type in registry, f"Registry should have {artifact_type}"
        print(f"{artifact_type}: {len(registry[artifact_type])} artifacts")


if __name__ == "__main__":
    test_registry_bootstrap()
    test_registry_resolve()
    test_get_modified_timestamp()
    test_poke()
    test_poke_triggers_detail_refresh()
    test_poke_single_stale_cycle()
    test_deleted_artifact()
    test_registry_full_dump()
    print("\n\nAll CachedMeta tests passed!")
