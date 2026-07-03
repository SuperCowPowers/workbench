# Feature Endpoints

SMILES-based molecular descriptor endpoints deployed on AWS SageMaker via Workbench.

## Endpoints

| Script | Endpoint Name | Description |
|--------|--------------|-------------|
| `smiles_to_2d_v1.py` | `smiles-to-2d-v1` | RDKit + Mordred 2D descriptors (salts removed) |
| `smiles_to_2d_keep_salts_v1.py` | `smiles-to-2d-keep-salts-v1` | RDKit + Mordred 2D descriptors (salts kept) |
| `smiles_to_3d_v1.py` | `smiles-to-3d-v1` | 3D conformer-based descriptors, async — 50-500 adaptive conformers, Boltzmann-weighted (74 features) |

## Deployment

Run from the `feature_endpoints/` directory:

```bash
# 2D Descriptors (salts removed) --> endpoint: smiles-to-2d-v1
python smiles_to_2d_v1.py

# 2D Descriptors (salts kept) --> endpoint: smiles-to-2d-keep-salts-v1
python smiles_to_2d_keep_salts_v1.py

# 3D Full (async) --> endpoint: smiles-to-3d-v1
python smiles_to_3d_v1.py

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
