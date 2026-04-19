# Feature Endpoints

SMILES-based molecular descriptor endpoints deployed on AWS SageMaker via Workbench.

## Endpoints

| Script | Endpoint Name | Description |
|--------|--------------|-------------|
| `smiles_to_2d_v1.py` | `smiles-to-2d-v1` | RDKit + Mordred 2D descriptors (salts removed) |
| `smiles_to_2d_keep_salts_v1.py` | `smiles-to-2d-keep-salts-v1` | RDKit + Mordred 2D descriptors (salts kept) |
| `smiles_to_3d_fast_v1.py` | `smiles-to-3d-fast-v1` | 3D conformer-based descriptors, fast realtime mode — 10 conformers (74 features) |
| `smiles_to_3d_full_v1.py` | `smiles-to-3d-full-v1` | 3D conformer-based descriptors, full async mode — 50-500 adaptive conformers (74 features, same set) |

## Deployment

Run from the `feature_endpoints/` directory:

```bash
# 2D Descriptors (salts removed) --> endpoint: smiles-to-2d-v1
python smiles_to_2d_v1.py

# 2D Descriptors (salts kept) --> endpoint: smiles-to-2d-keep-salts-v1
python smiles_to_2d_keep_salts_v1.py

# 3D Fast (realtime) --> endpoint: smiles-to-3d-fast-v1
# Default instance is ml.c7i.xlarge. Override with INSTANCE=ml.c7i.2xlarge for more vCPUs.
# Batch size is auto-tuned by config: serverless=3, xlarge=5, 2xlarge=10
SERVERLESS=false python smiles_to_3d_fast_v1.py
SERVERLESS=false INSTANCE=ml.c7i.2xlarge python smiles_to_3d_fast_v1.py

# 3D Full (async) --> endpoint: smiles-to-3d-full-v1
python smiles_to_3d_full_v1.py

# 2D endpoints support serverless or dedicated instance:
SERVERLESS=false python smiles_to_2d_v1.py
```

Each script will:
1. Create the `feature_endpoint_fs` FeatureSet (if it doesn't exist)
2. Build the model with its custom script
3. Deploy the SageMaker endpoint
4. Run a small test inference
