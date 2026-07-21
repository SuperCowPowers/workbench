# Exploring What Exists

> find what already exists in the account; health tags

Start here before building anything — most questions are answered by what's
already in the account.

## Inventory

```python
from workbench.cached.cached_meta import CachedMeta

meta = CachedMeta()
meta.models()               # fast, partial columns
meta.models(details=True)   # all columns, slower
```

The column names, so you don't have to guess (`details=True` changes which are
**populated**, not which exist — the set is the same either way):

```
models()     Model Group, Health, Owner, Type, Framework, Created, Modified,
             Ver, Input, Status, Description, Tags
endpoints()  Name, Health, Owner, Instance, Created, Modified, Input, Status,
             Config, Variant, Capture, Samp(%), Tags, Monitored
```

Note it is `Type`, not "Model Type", and `Model Group`, not "Name", for models —
endpoints use `Name`. Print `df.columns.tolist()` if unsure rather than guessing.

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
model.get_health_tags()     # [] means healthy
```

The `Health` column from `models(details=True)` / `endpoints(details=True)` is
the same information. Don't read a blank Health cell as "unknown" or "not
checked" — read it as healthy.

## Drilling in

```python
fs = FeatureSet("aqsol_features")
fs.columns                # property, not a method
fs.column_details()      # names + types
fs.pull_dataframe()      # the whole thing -- check row count first
fs.query("SELECT ... FROM ...")   # Athena, for anything large
```

Prefer `query()` over `pull_dataframe()` when the FeatureSet is big; pulling a
million rows into the REPL to compute a mean wastes minutes.

## Models

```python
model = Model("aqsol-reg")
model.details()
model.list_inference_runs()                    # capture names on this model
model.get_inference_metrics(capture)           # metrics for a capture (None if it has none)
model.get_inference_predictions(capture)       # predictions, populated after inference runs
```

`get_inference_metrics(capture)` returns `None` when the capture has no usable
metrics. A `No columns to parse from file` log means the metrics file is **empty,
not access-denied** (permissions would be a 403, raised before parsing). Read
`None` as "no metrics here," report it, and move on — don't retry.

## Reading the pipeline backwards

```python
model.get_input()    # the FeatureSet that trained this model
fs.get_input()   # the DataSource behind the FeatureSet
```

Useful when a model's provenance matters — which is most of the time.
