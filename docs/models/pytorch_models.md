# PyTorch Models

Workbench PyTorch models train feedforward neural networks on RDKit molecular descriptors. They support both regression and classification with built-in uncertainty quantification via cross-validation ensembles.

## Creating a PyTorch Model

```python
from workbench.api import FeatureSet, ModelType, ModelFramework

fs = FeatureSet("aqsol_features")

# Regression with uncertainty quantification
model = fs.to_model(
    name="sol-pytorch-reg",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.PYTORCH,
    target_column="solubility",
    feature_list=fs.feature_columns,
    description="PyTorch regression for solubility",
    tags=["pytorch", "solubility"],
)

# Deploy and run inference
endpoint = model.to_endpoint()
endpoint.auto_inference()
```

### Classification

```python
model = fs.to_model(
    name="sol-pytorch-class",
    model_type=ModelType.CLASSIFIER,
    model_framework=ModelFramework.PYTORCH,
    target_column="solubility_class",
    feature_list=fs.feature_columns,
    description="PyTorch classifier for solubility",
    tags=["pytorch", "classification"],
)
model.set_class_labels(["low", "medium", "high"])
```

## Hyperparameters

Set hyperparameters via the `hyperparameters` dict:

```python
model = fs.to_model(
    name="sol-pytorch-tuned",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.PYTORCH,
    target_column="solubility",
    feature_list=fs.feature_columns,
    hyperparameters={
        "layers": "256-128-64",
        "max_epochs": 200,
        "learning_rate": 0.001,
        "dropout": 0.1,
        "batch_size": 32,
    },
)
```

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Parameter</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Default</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">layers</td>
      <td style="padding: 8px 16px;">512-256-128</td>
      <td style="padding: 8px 16px;">Hidden layer sizes (dash-separated)</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">max_epochs</td>
      <td style="padding: 8px 16px;">200</td>
      <td style="padding: 8px 16px;">Maximum training epochs</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">learning_rate</td>
      <td style="padding: 8px 16px;">0.001</td>
      <td style="padding: 8px 16px;">Optimizer learning rate</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">dropout</td>
      <td style="padding: 8px 16px;">0.05</td>
      <td style="padding: 8px 16px;">Dropout rate</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">batch_size</td>
      <td style="padding: 8px 16px;">64</td>
      <td style="padding: 8px 16px;">Training batch size</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">early_stopping_patience</td>
      <td style="padding: 8px 16px;">30</td>
      <td style="padding: 8px 16px;">Epochs without improvement before stopping</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">loss</td>
      <td style="padding: 8px 16px;">L1Loss</td>
      <td style="padding: 8px 16px;">Loss function: L1Loss, MSELoss, HuberLoss, SmoothL1Loss</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">n_folds</td>
      <td style="padding: 8px 16px;">5</td>
      <td style="padding: 8px 16px;">Number of cross-validation folds (ensemble size)</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">split_strategy</td>
      <td style="padding: 8px 16px;">random</td>
      <td style="padding: 8px 16px;">Data splitting: random, scaffold, or butina</td>
    </tr>
  </tbody>
</table>

### Layer Architecture

The `layers` parameter defines the feedforward network architecture as a dash-separated string. Each number is a hidden layer dimension:

- `"512-256-128"` — Three hidden layers (default, good for most datasets)
- `"128-64-32"` — Smaller network for smaller datasets
- `"1024-512-256-128"` — Deeper network for large, complex datasets

!!! note "Examples"
    Full code listing: [`examples/models/pytorch.py`](https://github.com/SuperCowPowers/workbench/blob/main/examples/models/pytorch.py)

---

## Questions?

<img align="right" src="/images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)
