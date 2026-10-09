# Feature Endpoints

SMILES-based molecular feature endpoints (descriptors, fingerprints, embeddings) deployed
on AWS SageMaker via Workbench.

## Endpoints

| Script | Endpoint Name | Description |
|--------|--------------|-------------|
| `smiles_to_2d_v1.py` | `smiles-to-2d-v1` | RDKit + Mordred 2D descriptors (salts removed) |
| `smiles_to_2d_salt_v1.py` | `smiles-to-2d-salt-v1` | RDKit + Mordred 2D descriptors (salts kept) |
| `smiles_to_fingerprints_v1.py` | `smiles-to-fingerprints-v1` | Morgan count fingerprints (4096-dim, radius 2 / ECFP4) |
| `smiles_to_3d_v2.py` | `smiles-to-3d-v2` | Curated GFN2-xTB 3D descriptors, async (26 features) |
| `smiles_to_monroe_v1.py` | `smiles-to-monroe-v1` | Monroe pretrained molecular embedding (720 features, salts removed) |

MetaEndpoints fan out to both children and concatenate in one call:

| Script | Endpoint Name | Children |
|--------|--------------|----------|
| `smiles_to_2d_3d_v2.py` | `smiles-to-2d-3d-v2` | `smiles-to-2d-v1` + `smiles-to-3d-v2` |
| `smiles_to_2d_3d_salt_v2.py` | `smiles-to-2d-3d-salt-v2` | `smiles-to-2d-salt-v1` + `smiles-to-3d-v2` |

Salt-keeping is for **solubility only**, where the counterion is part of what was
measured. Every other assay uses the salt-removing endpoints.

### Deprecated

| Script | Endpoint Name | Description |
|--------|--------------|-------------|
| `smiles_to_3d_v1.py` | `smiles-to-3d-v1` | First-gen 3D set, async — 50-200 adaptive conformers, Boltzmann-weighted (74 features) |
| `smiles_to_2d_3d_v1.py` | `smiles-to-2d-3d-v1` | `smiles-to-2d-v1` + `smiles-to-3d-v1` |

Still deployed so existing models keep working and so the two 3D sets can be ablated
against each other. Not for new work — see `docs/blogs/3d_descriptors.md`.

## Deployment

Run from the `feature_endpoints/` directory:

```bash
# 2D Descriptors (salts removed) --> endpoint: smiles-to-2d-v1
python smiles_to_2d_v1.py

# 2D Descriptors (salts kept) --> endpoint: smiles-to-2d-salt-v1
python smiles_to_2d_salt_v1.py

# Morgan count fingerprints --> endpoint: smiles-to-fingerprints-v1
python smiles_to_fingerprints_v1.py

# 3D Full (async) --> endpoint: smiles-to-3d-v1
python smiles_to_3d_v1.py

# 3D Curated xTB (async) --> endpoint: smiles-to-3d-v2
python smiles_to_3d_v2.py

# MetaEndpoint, 2D + curated 3D --> endpoint: smiles-to-2d-3d-v2
python smiles_to_2d_3d_v2.py

# Monroe embedding --> endpoint: smiles-to-monroe-v1
python smiles_to_monroe_v1.py

# 2D endpoints support serverless or dedicated instance:
SERVERLESS=false python smiles_to_2d_v1.py
```

Each script will:
1. Create the `feature_endpoint_fs` FeatureSet (if it doesn't exist)
2. Build the model with its custom script
3. Deploy the SageMaker endpoint
4. Run a small test inference

## Monroe embedding

`smiles-to-monroe-v1` runs the frozen [Monroe](https://github.com/blazejba/monroe)
encoder (MIT) and returns the 720-d embedding as one compressed feature column,
`monroe` (comma-separated floats, like the fingerprint endpoint's `fingerprint`). Each
molecule is standardized, given one seeded RDKit conformer, and embedded; a molecule that
cannot be featurized keeps its row with NaN in `monroe`. When conformer generation fails
or passes its 10 s limit, the molecule is embedded from a flat 2D layout instead, and
nothing in the output marks that row.

- A FeatureSet built from it marks the column with `set_compressed_features(["monroe"])`;
  the model templates (XGBoost, PyTorch, TabICL) then expand it into 720 float columns.

- The encoder and featurizer are vendored in `model_scripts/monroe/` and formatted to the
  repo's lint rules; the functional changes from upstream are the lines marked `Workbench:`.
- The weights are a registered foundation checkpoint, read from Workbench's public bucket,
  that the model's training step copies into the model artifact. No setup per account.
- The model is `ModelType.TRANSFORMER` + `ModelFramework.PYTORCH`, which selects the
  `pytorch_chem` images (the encoder needs torch and `torch-geometric`).

## Autoscaling

| Deployment | Scaling |
|------------|---------|
| Serverless | AWS-managed via `max_concurrency` (scale to zero when idle) |
| Realtime (`SERVERLESS=false`) | Fixed at 1 instance, unless `max_instances` is set |
| Async (`smiles-to-3d-v1`) | Step-scales `0 → 8` on queue backlog |

Realtime endpoints default to a single fixed instance. Only `smiles_to_2d_v1.py`
opts into scaling (`MAX_INSTANCES=4`), since it's hit by many batch jobs at once;
it autoscales `1 → MAX_INSTANCES` on CPU (~60% variant-average — featurizers are
CPU-bound).
