# XGBoost Models

XGBoost is the default model framework in Workbench. It trains gradient boosted trees on RDKit molecular descriptors and supports both regression and classification with built-in uncertainty quantification.

## Creating an XGBoost Model

```python
from workbench.api import FeatureSet, ModelType

fs = FeatureSet("aqsol_features")

# Regression with uncertainty quantification
model = fs.to_model(
    name="sol-xgb-reg",
    model_type=ModelType.UQ_REGRESSOR,
    target_column="solubility",
    feature_list=fs.feature_columns,
    description="XGBoost regression for solubility",
    tags=["xgboost", "solubility"],
)

# Deploy and run inference
endpoint = model.to_endpoint()
endpoint.auto_inference()
```

### Classification

```python
model = fs.to_model(
    name="sol-xgb-class",
    model_type=ModelType.CLASSIFIER,
    target_column="solubility_class",
    feature_list=fs.feature_columns,
    description="XGBoost classifier for solubility",
    tags=["xgboost", "classification"],
)
model.set_class_labels(["low", "medium", "high"])
```

## Hyperparameters

Set hyperparameters via the `hyperparameters` dict:

```python
model = fs.to_model(
    name="sol-xgb-tuned",
    model_type=ModelType.UQ_REGRESSOR,
    target_column="solubility",
    feature_list=fs.feature_columns,
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
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
      <td style="padding: 8px 16px; font-weight: bold;">n_estimators</td>
      <td style="padding: 8px 16px;">300</td>
      <td style="padding: 8px 16px;">Number of boosted trees</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">max_depth</td>
      <td style="padding: 8px 16px;">7</td>
      <td style="padding: 8px 16px;">Maximum tree depth</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">learning_rate</td>
      <td style="padding: 8px 16px;">0.05</td>
      <td style="padding: 8px 16px;">Boosting learning rate (eta)</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">subsample</td>
      <td style="padding: 8px 16px;">0.8</td>
      <td style="padding: 8px 16px;">Fraction of training samples per tree</td>
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
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">butina_cutoff</td>
      <td style="padding: 8px 16px;">0.4</td>
      <td style="padding: 8px 16px;">Tanimoto distance threshold (butina splits only)</td>
    </tr>
  </tbody>
</table>

Any additional XGBoost parameters (e.g., `colsample_bytree`, `gamma`, `min_child_weight`) are passed directly to the XGBoost estimator.

!!! note "Examples"
    Full code listing: [`examples/models/xgb_model.py`](https://github.com/SuperCowPowers/workbench/blob/main/examples/models/xgb_model.py)

---

## Questions?

<img align="right" src="/images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)
