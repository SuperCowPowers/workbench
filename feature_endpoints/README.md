# Feature Endpoints

SMILES-based molecular descriptor endpoints deployed on AWS SageMaker via Workbench.

## Endpoints

| Script | Endpoint Name | Description |
|--------|--------------|-------------|
| `smiles_to_2d_v1.py` | `smiles-to-2d-v1` | RDKit + Mordred 2D descriptors (salts removed) |
| `smiles_to_2d_salt_v1.py` | `smiles-to-2d-salt-v1` | RDKit + Mordred 2D descriptors (salts kept) |
| `smiles_to_fingerprints_v1.py` | `smiles-to-fingerprints-v1` | Morgan count fingerprints (4096-dim, radius 2 / ECFP4) |
| `smiles_to_3d_v2.py` | `smiles-to-3d-v2` | Curated GFN2-xTB 3D descriptors, async (26 features) |

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

# 2D endpoints support serverless or dedicated instance:
SERVERLESS=false python smiles_to_2d_v1.py
```

Each script will:
1. Create the `feature_endpoint_fs` FeatureSet (if it doesn't exist)
2. Build the model with its custom script
3. Deploy the SageMaker endpoint
4. Run a small test inference

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
