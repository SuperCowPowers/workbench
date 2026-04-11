# Feature Endpoints

SMILES-based molecular descriptor endpoints deployed on AWS SageMaker via Workbench.

## Endpoints

| Script | Endpoint Name | Description |
|--------|--------------|-------------|
| `rdkit_mordred_v1.py` | `smiles-to-taut-md-stereo-v1` | RDKit + Mordred 2D descriptors (salts removed) |
| `rdkit_mordred_keep_salts_v1.py` | `smiles-to-taut-md-stereo-v1-keep-salts` | RDKit + Mordred 2D descriptors (salts kept) |
| `rdkit_3d_v1.py` | `smiles-to-3d-descriptors-v1` | 3D conformer-based descriptors (75 features) |

## Deployment

Run from the `feature_endpoints/` directory:

```bash
# 2D Descriptors (salts removed) --> endpoint: smiles-to-taut-md-stereo-v1
python rdkit_mordred_v1.py

# 2D Descriptors (salts kept) --> endpoint: smiles-to-taut-md-stereo-v1-keep-salts
python rdkit_mordred_keep_salts_v1.py

# 3D Descriptors --> endpoint: smiles-to-3d-descriptors-v1
# Default instance is ml.c7i.xlarge. Override with INSTANCE=ml.c7i.2xlarge for more vCPUs.
# Batch size is auto-tuned by config: serverless=3, xlarge=5, 2xlarge=10
SERVERLESS=false python rdkit_3d_v1.py
SERVERLESS=false INSTANCE=ml.c7i.2xlarge python rdkit_3d_v1.py

# 2D endpoints support serverless or dedicated instance:
SERVERLESS=false python rdkit_mordred_v1.py
```

Each script will:
1. Create the `feature_endpoint_fs` FeatureSet (if it doesn't exist)
2. Build the model with its custom script
3. Deploy the SageMaker endpoint
4. Run a small test inference
