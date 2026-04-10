# InferenceCache

`InferenceCache` is a caching wrapper around a Workbench `Endpoint`. It's handy when an endpoint is slow to invoke and the same inputs show up across calls — the motivating example is the 3D molecular feature endpoint `smiles-to-3d-descriptors-v1`, which takes real time to generate conformers and force-field optimize each molecule.

On each `inference(df)` call, rows whose cache-key value is already in the cache are served from S3, and only the new rows go to the underlying endpoint. Newly-computed rows are written back to the cache. The cache lives in a shared S3-backed `DFStore`, so once one person has computed a row, everyone gets it for free.

!!! note "Not the same as `workbench.cached.CachedEndpoint`"
    `CachedEndpoint` caches **metadata** methods like `summary()`, `details()`, and `health_check()`. `InferenceCache` caches **inference results**. Different classes, different concerns.

## Example

```py title="inference_cache_example.py"
from workbench.api import Endpoint, FeatureSet, InferenceCache

# Wrap a slow endpoint in an InferenceCache
endpoint = Endpoint("smiles-to-3d-descriptors-v1")
cached = InferenceCache(endpoint, cache_key_column="smiles")

# Pull a DataFrame of molecules and run inference
df = FeatureSet("feature_endpoint_fs").pull_dataframe()[:50]

# First call: slow (cache is empty, rows go to the endpoint)
results = cached.inference(df)

# Second call with the same SMILES: near-instant (all hits)
results_again = cached.inference(df)

# Drop a bad row so it recomputes on the next call
cached.delete_entries("c1ccc(cc1)C(=O)O")

# Or drop many at once
cached.delete_entries(["CCO", "CCN", "CCOCC"])

# Inspect the cache
print(cached.cache_size())
print(cached.cache_info())
```

**Output** (log lines)

```
InferenceCache[smiles-to-3d-descriptors-v1]: 0/50 cache hits
InferenceCache[smiles-to-3d-descriptors-v1]: computing 50 new rows via endpoint
InferenceCache[smiles-to-3d-descriptors-v1]: 50/50 cache hits
InferenceCache[smiles-to-3d-descriptors-v1]: removed 1 entries
InferenceCache[smiles-to-3d-descriptors-v1]: removed 3 entries
```

## Endpoint change detection

If you redeploy the underlying endpoint, `InferenceCache` notices. A tiny sidecar manifest stores the endpoint's `modified()` timestamp; on the next cache access, if it doesn't match the endpoint's current `modified()`, the cache is cleared automatically so you don't get stale results.

!!! note "Attribute delegation"
    `InferenceCache` forwards anything it doesn't define to the wrapped endpoint, so `cached.name`, `cached.details()`, `cached.fast_inference()`, etc. all Just Work.

## API Reference

::: workbench.api.inference_cache
