# InferenceCache

`InferenceCache` is a caching wrapper around a Workbench `Endpoint`. It's handy when an endpoint is slow to invoke and the same inputs show up across calls — the motivating example is the 3D molecular feature endpoint `smiles-to-3d-fast-v1`, which takes real time to generate conformers and force-field optimize each molecule.

On each `inference(df)` call, rows whose cache-key value is already in the cache are served from S3, and only the new rows go to the underlying endpoint. Newly-computed rows are written back to the cache. The cache lives in a shared S3-backed `DFStore`, so once one person has computed a row, everyone gets it for free.

!!! note "Not the same as `workbench.cached.CachedEndpoint`"
    `CachedEndpoint` caches **metadata** methods like `summary()`, `details()`, and `health_check()`. `InferenceCache` caches **inference results**. Different classes, different concerns.

## Example

```py title="inference_cache_example.py"
from workbench.api import Endpoint, FeatureSet, InferenceCache

# Wrap a slow endpoint in an InferenceCache
endpoint = Endpoint("smiles-to-3d-fast-v1")
cached_endpoint = InferenceCache(endpoint, cache_key_column="smiles")

# Pull a DataFrame of molecules and run inference
df = FeatureSet("feature_endpoint_fs").pull_dataframe()[:50]

# First call: slow (cache is empty, rows go to the endpoint)
results = cached_endpoint.inference(df)

# Second call with the same SMILES: near-instant (all hits)
results_again = cached_endpoint.inference(df)

# Drop a bad row so it recomputes on the next call
cached_endpoint.delete_entries("c1ccc(cc1)C(=O)O")

# Or drop many at once
cached_endpoint.delete_entries(["CCO", "CCN", "CCOCC"])

# Inspect the cache
print(cached_endpoint.cache_size())
print(cached_endpoint.cache_info())
```

**Output** (log lines)

```
InferenceCache[smiles-to-3d-fast-v1]: 0/50 cache hits
InferenceCache[smiles-to-3d-fast-v1]: computing 50 new rows via endpoint
InferenceCache[smiles-to-3d-fast-v1]: 50/50 cache hits
InferenceCache[smiles-to-3d-fast-v1]: removed 1 entries
InferenceCache[smiles-to-3d-fast-v1]: removed 3 entries
```

## Endpoint change detection

By default, `InferenceCache` keeps the existing cache regardless of endpoint changes. If you want it to automatically clear the cache when the endpoint has been modified since the cache was last written, pass `auto_invalidate_cache=True`:

```python
cached_endpoint = InferenceCache(endpoint, cache_key_column="smiles", auto_invalidate_cache=True)
```

A tiny sidecar manifest stores the endpoint's `modified()` timestamp; when auto-invalidation is enabled, the cache is cleared on the next access if the stored and current timestamps differ.

!!! note "Attribute delegation"
    `InferenceCache` forwards anything it doesn't define to the wrapped endpoint, so `cached_endpoint.name`, `cached_endpoint.details()`, `cached_endpoint.fast_inference()`, etc. all Just Work.

## API Reference

::: workbench.api.inference_cache
