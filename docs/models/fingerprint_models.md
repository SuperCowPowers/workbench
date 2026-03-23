# Fingerprint Models

Workbench supports training models on Morgan count fingerprints — a powerful molecular representation that captures substructure occurrence counts. Fingerprint models work with any framework (XGBoost, PyTorch) and can be combined with RDKit descriptors.

## Pipeline Overview

Fingerprint models use a two-step pipeline:

1. **Generate fingerprints** — A transformer endpoint computes Morgan count fingerprints from SMILES
2. **Train a model** — Use the `"fingerprint"` column as a feature (stored as a compressed column)


### Create the Fingerprint Endpoint
Note: This 'feature endpoint' only needs to be created once. It can then be reused across multiple models and FeatureSets. Please see our blog post [Feature Endpoints: Reusable Data Transformations](../blogs/feature_endpoints.md) for more details.

```python
from workbench.api import FeatureSet, ModelType
from workbench.utils.model_utils import get_custom_script_path

tags = ["smiles", "morgan fingerprints"]
script_path = get_custom_script_path("chem_info", "morgan_fingerprints.py")
feature_set = FeatureSet("aqsol_features")
model = feature_set.to_model(
    name="smiles-to-fingerprints-v0",
    model_type=ModelType.TRANSFORMER,
    feature_list=["smiles"],
    description="Smiles to Morgan Fingerprints",
    tags=tags,
    custom_script=script_path,
)

# Create the endpoint for the model
end = model.to_endpoint(tags=tags)
end.auto_inference()
```

### Step 1: Compute Fingerprints and Create a FeatureSet

Run your data through the fingerprint endpoint, then create a FeatureSet and mark the fingerprint column as compressed:

```python
from workbench.api import DataSource, Endpoint
from workbench.core.transforms.pandas_transforms import PandasToFeatures

ds = DataSource("aqsol_data")
df = ds.pull_dataframe()

# Run the data through our Smiles to Fingerprints Endpoint
fp_end = Endpoint("smiles-to-fingerprints-v0")
df_with_fp = fp_end.inference(df)

# Create a Feature Set
to_features = PandasToFeatures("aqsol_fingerprints")
to_features.set_input(df_with_fp, id_column="id")
to_features.set_output_tags(["aqsol", "fingerprints"])
to_features.transform()

# Set our compressed features for this FeatureSet
fs = FeatureSet("aqsol_fingerprints")
fs.set_compressed_features(["fingerprint"])
```

### Step 2: Train a Model on Fingerprints

```python
from workbench.api import FeatureSet, ModelType

fs = FeatureSet("aqsol_fingerprints")

model = fs.to_model(
    name="aqsol-fingerprint-reg-v0",
    model_type=ModelType.UQ_REGRESSOR,
    target_column="solubility",
    feature_list=["fingerprint"],
    description="Model for Aqueous Solubility using Morgan Fingerprints",
    tags=["aqsol", "fingerprints", "regression"],
)
end = model.to_endpoint(tags=["aqsol", "fingerprints", "regression"])
end.auto_inference()
```

## Fingerprints + Descriptors

Combine fingerprints with RDKit molecular descriptors for richer feature sets:

```python
# Grab features from an existing descriptor model and add fingerprints
descriptor_features = Model("sol-xgb-reg").features()
combined_features = descriptor_features + ["fingerprint"]

model = fs.to_model(
    name="sol-fingerprints-plus",
    model_type=ModelType.REGRESSOR,
    target_column="solubility",
    feature_list=combined_features,
    description="Fingerprints + Molecular Descriptors",
    tags=["fingerprints", "descriptors"],
)
```

## Using PyTorch with Fingerprints

```python
from workbench.api import FeatureSet, ModelType, ModelFramework

fs = FeatureSet("aqsol_fingerprints")

model = fs.to_model(
    name="sol-fp-pytorch",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.PYTORCH,
    target_column="solubility",
    feature_list=["fingerprint"],
    description="PyTorch model on Morgan Fingerprints",
    tags=["fingerprints", "pytorch"],
)
```

## How Count Fingerprints Work

Morgan count fingerprints (ECFP4 equivalent) encode substructure occurrence counts rather than binary presence/absence:

- **Radius 2** (ECFP4) — captures local chemical environments up to 2 bonds from each atom
- **2048 bits** — hashed into a fixed-length vector
- **Count values (0–255)** — how many times each substructure occurs, providing richer information than binary fingerprints

!!! note "Examples"
    Full code listing: [`examples/models/smiles_to_fingerprints.py`](https://github.com/SuperCowPowers/workbench/blob/main/examples/models/smiles_to_fingerprints.py)

---

## Questions?

<img align="right" src="/images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)
