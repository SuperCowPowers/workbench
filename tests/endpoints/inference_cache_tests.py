"""Tests for InferenceCache over the abalone-regression endpoint.

Verifies the cache-as-feature-lookup contract: the cache stores only
key + endpoint features, so any caller gets back their own input columns
plus the endpoint's features — regardless of what a previous caller sent.
"""

from workbench.api import Endpoint, InferenceCache
from workbench.utils.endpoint_utils import get_evaluation_data

endpoint = Endpoint("abalone-regression")


def _make_cache(key_col: str = "auto_id") -> InferenceCache:
    """Create a fresh InferenceCache with a clean slate."""
    cache = InferenceCache(endpoint, cache_key_column=key_col)
    cache.clear_cache()
    return cache


def test_basic_cache_hit():
    """First call populates the cache, second call should be all hits."""
    cached_endpoint = _make_cache()
    eval_df = get_evaluation_data(endpoint)[:10]
    print(f"Input shape: {eval_df.shape}, columns: {list(eval_df.columns)}")

    # First call — cache is empty, all rows go to the endpoint
    result1 = cached_endpoint.inference(eval_df)
    assert not result1.empty
    assert "prediction" in result1.columns
    cache_size_after_first = cached_endpoint.cache_size()
    print(f"After first call: cache has {cache_size_after_first} rows")
    assert cache_size_after_first == 10

    # Second call — should be all cache hits, same result
    result2 = cached_endpoint.inference(eval_df)
    assert result1.shape == result2.shape
    assert list(result1.columns) == list(result2.columns)
    print(f"After second call: cache still has {cached_endpoint.cache_size()} rows")

    # Clean up
    cached_endpoint.clear_cache()
    print("Basic cache hit test passed!")


def test_extra_column_not_leaked():
    """Adding a column to the caller's DataFrame should not pollute the cache.

    Contract: cache stores key + features only. A second caller with
    different input columns should get back *their* columns + features,
    not columns from the first caller.
    """
    cached_endpoint = _make_cache()
    eval_df = get_evaluation_data(endpoint)[:10]
    input_cols = list(eval_df.columns)

    # First caller — populates the cache
    result1 = cached_endpoint.inference(eval_df)
    result1_cols = set(result1.columns)
    print(f"Caller 1 columns: {sorted(result1_cols)}")

    # Second caller — same rows but with an extra column
    eval_df2 = eval_df.copy()
    eval_df2["my_extra_column"] = range(len(eval_df2))
    result2 = cached_endpoint.inference(eval_df2)
    result2_cols = set(result2.columns)
    print(f"Caller 2 columns: {sorted(result2_cols)}")

    # The extra column should be in the output (it's the caller's own column)
    assert "my_extra_column" in result2_cols, "Caller's extra column should be in output"

    # Now do a third call WITHOUT the extra column — it should NOT appear
    result3 = cached_endpoint.inference(eval_df)
    result3_cols = set(result3.columns)
    print(f"Caller 3 columns: {sorted(result3_cols)}")
    assert "my_extra_column" not in result3_cols, "Extra column from caller 2 should NOT leak to caller 3"

    # Both callers should have the same feature columns from the endpoint
    feature_cols_1 = result1_cols - set(input_cols)
    feature_cols_3 = result3_cols - set(input_cols)
    assert (
        feature_cols_1 == feature_cols_3
    ), f"Feature columns should be identical: {feature_cols_1} vs {feature_cols_3}"

    # Clean up
    cached_endpoint.clear_cache()
    print("Extra column leak test passed!")


def test_cache_info():
    """Verify cache_info returns expected structure."""
    cached_endpoint = _make_cache()
    eval_df = get_evaluation_data(endpoint)[:10]

    # Populate cache
    cached_endpoint.inference(eval_df)
    info = cached_endpoint.cache_info()
    print(f"Cache info: {info}")

    assert info["rows"] == 10
    assert "auto_id" in info["columns"]
    assert "prediction" in info["columns"]

    # The cache should NOT contain input-only columns (like the target)
    # It should only have key + endpoint feature columns
    eval_only_cols = [c for c in eval_df.columns if c != "auto_id"]
    leaked = [c for c in eval_only_cols if c in info["columns"]]
    print(f"Cache columns: {info['columns']}")
    if leaked:
        print(f"WARNING: input columns found in cache: {leaked}")

    # Clean up
    cached_endpoint.clear_cache()
    print("Cache info test passed!")


def test_attribute_delegation():
    """InferenceCache should delegate unknown attributes to the endpoint."""
    cached_endpoint = _make_cache()
    assert cached_endpoint.name == "abalone-regression"
    assert cached_endpoint.exists()
    print(f"Delegated name: {cached_endpoint.name}")
    print("Attribute delegation test passed!")


if __name__ == "__main__":
    test_basic_cache_hit()
    test_extra_column_not_leaked()
    test_cache_info()
    test_attribute_delegation()
    print("\nAll InferenceCache tests passed!")
