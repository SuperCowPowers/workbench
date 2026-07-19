# Exploring What Exists

Start here before building anything — most questions are answered by what's
already in the account.

## Inventory

```python
from workbench.cached.cached_meta import CachedMeta

meta = CachedMeta()
meta.models()               # fast, partial columns
meta.models(details=True)   # all columns, slower
```

The REPL also exposes these as bare commands:

```python
summary()        # everything, by artifact type
data_sources()
feature_sets()
models()
endpoints()
```

These return DataFrames. Filter them rather than eyeballing:

```python
models()[models()["Model Type"] == "regressor"]
```

## Health

Health tags are **exceptions, not status**. A healthy artifact has an empty
health tag list — no news is good news. Any tags present mean something is
wrong with that artifact, and the tags name the problem.

```python
m.get_health_tags()     # [] means healthy
```

The `Health` column from `models(details=True)` / `endpoints(details=True)` is
the same information. Don't read a blank Health cell as "unknown" or "not
checked" — read it as healthy.

## Drilling in

```python
fs = FeatureSet("aqsol_features")
fs.column_names()
fs.column_details()      # names + types
fs.pull_dataframe()      # the whole thing -- check row count first
fs.query("SELECT ... FROM ...")   # Athena, for anything large
```

Prefer `query()` over `pull_dataframe()` when the FeatureSet is big; pulling a
million rows into the REPL to compute a mean wastes minutes.

## Models

```python
m = Model("aqsol-reg")
m.details()
m.performance_metrics()
m.get_inference_predictions()   # populated after endpoint inference runs
```

## Reading the pipeline backwards

```python
m.get_input()    # the FeatureSet that trained this model
fs.get_input()   # the DataSource behind the FeatureSet
```

Useful when a model's provenance matters — which is most of the time.
