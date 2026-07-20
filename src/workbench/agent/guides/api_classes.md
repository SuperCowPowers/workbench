# The API Classes — Lego Pieces

> the core classes and how they compose into pipelines

Workbench is a small set of classes with **uniform connectors**. That is the
whole design: few shapes, predictable interfaces, so they compose instead of
locking you into one path.

```python
from workbench.api import (
    DataSource, FeatureSet, Model, ModelType, ModelFramework, Endpoint,
    MetaEndpoint, InferenceCache, Meta, ParameterStore, DFStore, Reports,
    InferenceStore, PublicData, Monitor,
)
```

## The chain

```
DataSource -> FeatureSet -> Model -> Endpoint
```

Each stage is a **first-class artifact in AWS**, not a step inside a job. So you
never have to replay the pipeline to work with one:

```python
model = Model("aqsol-regression")     # just pick it up by name
end = Endpoint("aqsol-regression")
```

This is the flexibility that matters day to day: enter at any stage, inspect
anything, rebuild only the piece that changed. A FeatureSet built last month
feeds a model you train today.

## Uniform artifact interface

Every artifact answers the same questions, so what you learn on one applies to
all:

```python
art.exists()
art.details()
art.get_tags()
art.get_input()        # the artifact upstream of this one
art.health_check()     # [] means healthy
art.delete()
```

DataSource and FeatureSet also share their data accessors — `columns`,
`column_details()`, `pull_dataframe()`, `query()`.

## DataFrame in, DataFrame out

The connector between pieces is a pandas DataFrame. Endpoints take one and
return it with columns appended, which is why they chain:

```python
df_features = Endpoint("smiles-to-2d-v1").inference(df)   # features appended
preds       = Endpoint("my-admet-model").inference(df_features)
```

Because the wire format is just pandas, anything that produces a DataFrame
(`PublicData.get()`, an Athena query, your own code) is a valid starting point.

## Uniform stores

The storage classes share one interface — `get`, `upsert`, `list`, `delete`,
`check`, `last_modified` — and differ only in payload:

| Class | Holds |
|---|---|
| `DFStore` | DataFrames (S3 parquet) |
| `ParameterStore` | small values / config |
| `GraphStore` | networkx graphs |
| `Reports` | published reports (a `DFStore` scoped to `/reports`) |
| `InferenceStore` | inference results, Athena-queryable |

Learn one, you know the rest. Use them as the glue for anything that isn't an
artifact — intermediate frames, config, cross-session state.

## Wrappers and composition

Several classes take a piece and give back something with the same shape, which
is what makes them stackable:

- **`MetaEndpoint`** — an Endpoint backed by a DAG of child endpoints plus
  aggregation nodes. Still an Endpoint: same `inference()` call. Used for both
  feature pipelines (2D + 3D in one call) and ensembles.
- **`InferenceCache`** — wraps an Endpoint's `inference()` with S3 caching.
  Same call, skips recompute. Worth it for expensive endpoints like 3D.
- **`Monitor`** — wraps a deployed Endpoint with data-capture and model-quality
  monitors.

## Discovery

```python
from workbench.cached.cached_meta import CachedMeta
meta = CachedMeta()      # models(), endpoints(), feature_sets(), pipelines()
```

`Meta`/`CachedMeta` is the index over everything above — use it to find what
exists before building something new.

## More

- API classes: https://supercowpowers.github.io/workbench/api_classes/overview/
