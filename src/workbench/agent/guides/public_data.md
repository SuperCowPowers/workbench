# Public Data

> public S3 sample datasets for demos and experiments

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

- `common/` — abalone, wine, test_data. Small, generic fixtures for smoke tests.
- `comp_chem/` — the real cheminformatics data: `aqsol` (solubility, plus
  `aqsol/alignment` subsets), `logp`, `logd`, `logp_logd` (overlap bins),
  `openadmet` (four challenges, below), `reference_compounds`,
  `synthetic/multi_task`, `compound_sets`.

Datasets that ship a train/test split put them in `training/` and `testing/`
subdirs, so `comp_chem/openadmet/<challenge>/training/<endpoint>` is the shape
to expect. The four challenges are `expansionrx` (9 ADMET endpoints, plus an
`all_endpoints` wide table for multi-task), `pxr` (induction), `asap`
(antiviral ADMET + potency), and `octant_cyp` (CYP3A4 inhibition/reactivity, no
split).

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

Every published dataset is described (source, license, per-column meanings), so
`None` means the name is wrong.

## Into the pipeline

`get()` hands back a DataFrame, so go straight to a FeatureSet — no DataSource in
between (see `data_and_features`):

```python
from workbench.core.transforms.pandas_transforms import PandasToFeatures

df = pub_data.get("comp_chem/aqsol/aqsol_public_data")
to_features = PandasToFeatures("aqsol_features")
to_features.set_input(df, id_column="id")
to_features.transform()
```

Check the columns before choosing an `id_column` — these are external datasets
and the id column is not always called `id`.
