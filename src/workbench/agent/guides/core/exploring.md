# Exploring What Exists

> find what already exists in the account; health tags

Start here before building anything — most questions are answered by what's
already in the account.

## Inventory

```python
from workbench.api import CachedMeta

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
models_df = models(details=True)
models_df[models_df["Type"] == "regressor"]
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
df = fs.pull_dataframe()   # then pandas for everything else
```

Same call on a DataSource. See `data_and_features`.

## Models

```python
model = Model("aqsol-reg")
model.details()
model.hyperparameters()                        # how it was trained -- see below
model.list_inference_runs()                    # capture names on this model
model.get_inference_metrics(capture)           # metrics for a capture (None if it has none)
model.get_inference_predictions(capture)       # predictions, populated after inference runs
```

**`hyperparameters()` is how a model was trained** — it carries `split_strategy`
(`scaffold` by default, or `butina`/`random`), `butina_cutoff`, `n_folds`, `seed`,
and `uq_version`. Also includes the hyperparameters used for training.

`get_inference_metrics(capture)` returns `None` when thay capture doesn't exist.

## Reading the pipeline backwards

```python
model.get_input()    # the FeatureSet that trained this model
fs.get_input()   # the DataSource behind the FeatureSet
```

Useful when a model's provenance matters — which is most of the time.
