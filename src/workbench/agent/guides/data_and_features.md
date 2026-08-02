# DataSources and FeatureSets

> DataSources and FeatureSets: pulling data, columns, a model's training data

`DataSource -> FeatureSet` is the front half of the pipeline. A DataSource is the
landed data; a FeatureSet is the modeling-ready version with an id column and a
feature store behind it.

## DataSource to FeatureSet

Have a DataSource? `ds.to_features()`. Have a DataFrame? `PandasToFeatures()`

```python
ds = DataSource("s3://bucket/data.csv", name="my_data")   # or a local path, or a DataFrame
fs = ds.to_features("my_features", id_column="id", tags=["my_data"])
```

```python
from workbench.core.transforms.pandas_transforms import PandasToFeatures

to_features = PandasToFeatures("my_features")
to_features.set_input(df, id_column="id")
to_features.set_output_tags(["my_data"])
to_features.transform()
```

- Names must be lowercase, and a DataFrame needs an explicit `name` — there's no
  basename to derive one from.
- `id_column` is **required**. If the data has no natural id, say so rather than
  silently inventing a row index — the id is how predictions get joined back.
- `event_time_column` — set it if the data is temporal; otherwise one is
  generated.
- `one_hot_columns` — categoricals to expand at feature-creation time.

## Looking at the data — pull a DataFrame

```python
df = ds.pull_dataframe()      # DataSource
df = fs.pull_dataframe()      # FeatureSet -- same call
```

**This is how you look at the data.** A DataFrame gives you everything — types,
stats, counts, distributions, correlations, filtering, grouping. Use pandas
(`df.dtypes`, `df.describe()`, `df["target"].value_counts()`) rather than hunting
for a Workbench accessor that does the same thing.

`fs.columns` is a property (no parens) if you just want the names without a pull.

`fs.id_column` is the id, and it is authoritative — often not `"id"`. It is the
one join key for the whole chain: chosen from a DataSource column at
`to_features()`, used to build the model's training view, and resolved back off
the FeatureSet when an endpoint needs an id at inference. Read it, never guess it.

For a targeted pull, `query()` takes SQL. Use `fs.table` in the FROM clause — it
is the resolved Athena table, which carries a version suffix the FeatureSet name
does not:

```python
df = fs.query(f'SELECT smiles, logd FROM "{fs.table}" WHERE logd > 3')
```

## Column names are lowercase

This is **AWS behavior, not a Workbench choice**. Glue lowercases column names
when it creates a table, and Athena "accepts mixed case in DDL and DML queries,
but lower cases the names when it executes the query." A column created as
`Castle` comes back as `castle`.

Workbench lowercases on the way in so what you see matches what AWS will
actually store — otherwise the mismatch would surface later as a confusing
query failure.

Consequences:

- Anything read back from a DataSource or FeatureSet is lowercase.
- Raw external files are not — the public AqSol CSV has `SMILES`, client files
  vary.
- Case alone never distinguishes two columns. Don't hardcode a spelling; match
  case-insensitively when the source might be raw:

  ```python
  col = next(c for c in df.columns if c.lower() == "smiles")
  ```

## A model's input data

```python
fs = FeatureSet(model.get_input())        # the data itself
df = model.training_view().pull_dataframe()   # what training saw
```

The training view is that FeatureSet minus its excluded rows, plus
`sample_weight` and `validation`. Reach for it when the question is about
training — which rows were held out, weights, residuals per row. For everything about
the data — quality, distributions, cleanup — go to the FeatureSet, which still
has the excluded rows.

## Provenance

```python
fs.get_input()    # the DataSource behind this FeatureSet
```

Deleting an upstream artifact orphans everything downstream — see the
dependency chain in `making_models`.
