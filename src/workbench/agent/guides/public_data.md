# Public Data

Read-only sample datasets in a public S3 bucket (`workbench-public-data`).
Access is anonymous — no credentials, no AWS permissions needed — so this is a
safe way to get real data for demos, tests, and experiments.

The REPL exposes an instance as `pub_data`.

## Browse

```python
pub_data.list()       # dataset names
pub_data.details()    # DataFrame: name, size (MB), modified
```

Names are paths like `comp_chem/aqsol/aqsol_public_data`. The main groups:

- `common/` — abalone, wine. Small, generic, good for smoke tests.
- `comp_chem/` — the real cheminformatics data: `aqsol` (solubility), `logp`,
  `logd`, `openadmet_pxr`, `open_admet_expansionrx` (train/test pairs),
  `reference_compounds`, `synthetic/multi_task`.
- `testing/` — fixtures for tests.

Two quirks worth knowing: `list()` returns names **without** the `.csv`
extension while `details()` shows them **with** it, and `descriptions` appears
in `list()` but is the metadata file, not a dataset.

## Fetch

```python
df = pub_data.get("comp_chem/aqsol/aqsol_public_data")
```

Returns a DataFrame, or `None` with a warning if the name is unknown — check the
result rather than assuming it worked. Use the name exactly as `list()` gives
it (no `.csv`).

```python
pub_data.describe("comp_chem/logd/logd_all")   # source refs; None if undescribed
```

Not every dataset has a description — `None` is normal, not an error.

## Into the pipeline

A fetched DataFrame is an ordinary DataSource source, so the usual flow applies:

```python
df = pub_data.get("comp_chem/aqsol/aqsol_public_data")
ds = DataSource(df, name="aqsol_data")      # name required for a DataFrame
fs = ds.to_features("aqsol_features", id_column="id")
```

Check the columns before choosing an `id_column` — these are external datasets
and the id column is not always called `id`.
