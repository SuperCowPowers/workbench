# Exploring What Exists

Start here before building anything — most questions are answered by what's
already in the account.

## Inventory

Use `CachedMeta()` — it is much faster than `Meta()`. Only reach for `Meta()`
when the user explicitly wants live, uncached values.

```python
meta = CachedMeta()
meta.models()
```

`models()` and `endpoints()` take `details` (default `False`). The default is a
fast summary with only a **subset** of columns populated — the rest come back
empty. Pass `details=True` to fill every column (Health, Type, Framework,
metrics, ...); it is slower because it pulls per-artifact detail, so only reach
for it when you actually need those columns.

```python
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
