# Feature Endpoints

> SMILES-to-descriptor endpoints; avoiding train/inference skew

A feature endpoint is an Endpoint that computes **features instead of
predictions** — it holds no trained model. Model type is
`ModelType.TRANSFORMER`. Send it SMILES, get molecular descriptors back.

The point is eliminating training/inference skew: training, deployed inference,
batch jobs, and external platforms all call the *same* endpoint, so features are
identical by construction rather than by discipline. Two copies of descriptor
code drifting apart (an RDKit bump here, a Mordred edge case there) is a classic
source of silent model degradation.

## DataFrame in, DataFrame out

Every Workbench endpoint — feature and predictor alike — follows the same
contract: pass a DataFrame, get a DataFrame back. A feature endpoint returns
your rows with descriptor columns **appended**, so the output is a superset of
the input, not a replacement.

```python
end = Endpoint("smiles-to-2d-v1")
df_features = end.inference(df)        # df + ~315 descriptor columns
```

That contract is what lets you chain them: the output of a feature endpoint is
already the input a predictor endpoint expects.

```python
df_features = Endpoint("smiles-to-2d-v1").inference(input_df)
predictions  = Endpoint("my-admet-model").inference(df_features)
```

## Available endpoints

| Endpoint | Features | Use case |
|---|---|---|
| `smiles-to-2d` | ~315 2D descriptors | Standard ADMET modeling |
| `smiles-to-2d-keep-salts` | ~315 2D descriptors | Salt-sensitive work (solubility, formulation) |
| `smiles-to-3d-v1` | 74 3D descriptors | Full first-gen 3D set — async |
| `smiles-to-3d-v2` | 26 3D descriptors | Curated GFN2-xTB set, orthogonal to 2D — recommended, async |
| `smiles-to-fingerprints` | 2048-dim Morgan counts | Similarity, substructure models |

Combined **MetaEndpoints** fan out to both children and concatenate in one call:

- `smiles-to-2d-3d-v2` — 2D + curated 3D v2, ~339 features (prefer this)
- `smiles-to-2d-3d-v1` — 2D + full 3D v1, ~387 features

Use a MetaEndpoint rather than calling two endpoints and merging by hand.

The 3D endpoints are async and compute-intensive (conformer generation, xTB
energy ranking) — roughly 1-2 molecules/second. Expect real wall-clock time on
a large batch; that is the work, not a hang.

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

## Versioning

Versions are pinned by name (`-v1`, `-v2`), so a model keeps getting the
features it trained against while new models adopt an improved set. When a model
looks wrong, check which feature endpoint it was built against before suspecting
the model.

## More

- Feature endpoints: https://supercowpowers.github.io/workbench/blogs/feature_endpoints/
- MetaEndpoints: https://supercowpowers.github.io/workbench/models/meta_endpoints/
