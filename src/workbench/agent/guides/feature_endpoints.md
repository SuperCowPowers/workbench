# Feature Endpoints

> SMILES-to-feature endpoints; avoiding train/inference skew

A feature endpoint is an Endpoint that computes **features instead of
predictions** — it holds no trained model. Model type is
`ModelType.TRANSFORMER`. Send it SMILES, get molecular features back —
descriptors or fingerprints.

The point is eliminating training/inference skew: training, deployed inference,
batch jobs, and external platforms all call the *same* endpoint, so features are
identical by construction rather than by discipline. Two copies of descriptor
code drifting apart (an RDKit bump here, a Mordred edge case there) is a classic
source of silent model degradation.

## DataFrame in, DataFrame out

A feature endpoint appends descriptor columns, so the output is a superset of the
input — which is exactly the input a predictor endpoint expects:

```python
df_features = Endpoint("smiles-to-2d-v1").inference(input_df)   # + ~315 columns
predictions = Endpoint("my-admet-model").inference(df_features)
```

## Available endpoints

| Endpoint | Features | Use case |
|---|---|---|
| `smiles-to-2d-v1` | ~315 2D descriptors | **The default** for every assay |
| `smiles-to-fingerprints-v1` | 4096-dim Morgan counts | Similarity, substructure, fingerprint models |
| `smiles-to-3d-v2` | 26 3D descriptors | Curated GFN2-xTB set, orthogonal to 2D — async |
| `smiles-to-2d-salt-v1` | ~315 2D descriptors | Solubility only — see below |

Combined **MetaEndpoints** fan out to both children and concatenate in one call:

- `smiles-to-2d-3d-v2` — 2D + curated 3D v2, ~339 features (prefer this)
- `smiles-to-2d-3d-salt-v2` — the salt-keeping 2D paired with curated 3D v2

Use a MetaEndpoint rather than calling two endpoints and merging by hand.

**Deprecated:** `smiles-to-3d-v1` (74 features) and `smiles-to-2d-3d-v1`. Still deployed
so existing models keep working, and useful as an ablation baseline — never the choice
for new work.

**Keeping salts is the exception, not a judgment call.** Solubility is the assay
where the counterion is part of what was measured, so it gets `-salt-`. Every other
assay — permeability, clearance, CYP inhibition, binding — uses the salt-removing
default. Don't reach for the salt endpoint because the input happens to contain
salts; that is exactly what the default is built to strip.

## How to run one

`inference()` blocks for the whole pass, so the runner follows the wall clock. For
~5,000 molecules:

| Endpoint | Runtime | Run it as |
|---|---|---|
| 2D, fingerprints | ~10 minutes | a subprocess job — `run_feature_endpoint(df, name, wait=False)` |
| 3D, 2D+3D | 3-4 hours | a Batch script (`batch`), which warms the cache |

The 3D endpoints are compute-intensive (conformer generation, xTB energy ranking)
— roughly 1-2 molecules/second. That is the work, not a hang. Async is how
SageMaker runs the invocation; the caller still waits.

Both runners cache through `InferenceCache`, so a repeat pass over molecules already
seen returns in seconds. Details in `local_models`.

## What's in the pipeline

1. **Standardization** — salt extraction, charge neutralization, tautomer
   canonicalization
2. **RDKit** (~220) — constitutional, topological, electronic, lipophilicity
3. **Mordred** (~85) — AcidBase, Aromatic, Constitutional, Chi, CarbonTypes
4. **Stereochemistry** (10) — R/S and E/Z counts, stereo complexity

## Knowing the columns

```python
end.input_columns()      # what it consumes, e.g. ["smiles"]
end.output_columns()     # the registered feature columns
```

These are registered in ParameterStore, so downstream training scripts can look
up a feature list without calling the endpoint:

```python
from workbench.utils.endpoint_utils import get_output_columns
cols = get_output_columns("smiles-to-2d-v1")
```

Not every returned column is a feature. Provenance columns — `orig_smiles`,
`salt`, `undefined_chiral_centers` — come back alongside the descriptors and are
deliberately excluded from the registered output columns. Don't feed them into a
`feature_list`.

## Fingerprints: config lives in hyperparameters

`smiles-to-fingerprints-v1` produces Morgan **count** fingerprints at radius 2,
4096 bits (wide enough to limit count-corrupting collisions). Inference appends a
single `fingerprint` column — the 4096 counts packed as a comma-separated string,
not 4096 columns. Its featurization config is recorded as the model's
hyperparameters, so anything consuming a fingerprint model can resolve
radius/bits/counts rather than assuming them:

```python
model_name = end.get_input()                 # endpoint -> its input model
Model(model_name).hyperparameters()          # {'radius': 2, 'n_bits': 4096, 'counts': True}
```

A different radius/bits/counts mix is a new version (`-v2`), self-describing via
its own hyperparameters.

## Versioning

Versions are pinned by name (`-v1`, `-v2`), so a model keeps getting the
features it trained against while new models adopt an improved set. When a model
looks wrong, check which feature endpoint it was built against before suspecting
the model.

## Source

Deployment scripts and the authoritative endpoint list live in
`feature_endpoints/` (README + one script per endpoint). Check there to confirm
a name or see how a feature endpoint is built.

## More

- Feature endpoints: https://supercowpowers.github.io/workbench/blogs/feature_endpoints/
- MetaEndpoints: https://supercowpowers.github.io/workbench/models/meta_endpoints/
