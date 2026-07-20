# DataSources and FeatureSets

> DataSources and FeatureSets: creation, columns, queries, views

`DataSource -> FeatureSet` is the front half of the pipeline. A DataSource is
the landed, queryable data; a FeatureSet is the modeling-ready version with an
id column and a feature store behind it.

## Creating a DataSource

```python
ds = DataSource("s3://bucket/data.csv", name="my_data")
ds = DataSource("/local/path/data.csv", name="my_data")
ds = DataSource(df, name="my_data")        # name is REQUIRED for a DataFrame
```

Names must be lowercase. If you pass a DataFrame without a `name`, it raises —
there is no basename to derive one from.

## DataSource to FeatureSet

```python
fs = ds.to_features("my_features", id_column="id", tags=["my_data"])
```

- `id_column` is **required**. If the data has no natural id, say so rather than
  silently inventing a row index — the id is how predictions get joined back.
- `event_time_column` — set it if the data is temporal; otherwise one is
  generated.
- `one_hot_columns` — categoricals to expand at feature-creation time.

## Looking at the data

```python
ds.columns                 # property -- list of column names
ds.column_types            # property -- name -> type
ds.column_details()        # method  -- names + types together
ds.column_stats()          # method  -- per-column statistics
ds.num_rows, ds.num_columns
```

Note which are properties and which are methods; the same set exists on both
DataSource and FeatureSet. There is no `column_names()`.

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

## Getting rows

```python
df = fs.pull_dataframe()              # defaults to a 100k row limit
df = fs.query("SELECT ... FROM ...")  # Athena -- use for anything large
```

`pull_dataframe()` caps at 100,000 rows by default, so a large FeatureSet comes
back silently truncated. Check `fs.num_rows` first, and push filtering and
aggregation into `query()` rather than pulling everything to count or average
it in pandas.

## Views

A FeatureSet has views — different column subsets over the same rows.

```python
fs.views()                        # available view names
fs.view("training").pull_dataframe()
fs.set_display_columns([...])     # what the UI shows
fs.set_computation_columns([...]) # what gets computed on
```

The `training` view is what `to_model()` trains from, so inspect it when a
model's feature list looks wrong.

## Provenance

```python
fs.get_input()    # the DataSource behind this FeatureSet
```

Deleting an upstream artifact orphans everything downstream — see the
dependency chain in `making_models`.
