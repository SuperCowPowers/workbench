# Models

Workbench supports multiple model frameworks — all using the same API. Just change the `model_framework` parameter (or `model_class` for scikit-learn) and everything else stays the same: training, deployment, inference, and confidence scoring.

## Available Frameworks

| Framework | Description | Guide |
|-----------|-------------|-------|
| **XGBoost** | Gradient boosted trees on molecular descriptors | **[XGBoost Models](xgboost_models.md)** |
| **Scikit-Learn** | Any scikit-learn estimator (RandomForest, KMeans, etc.) | **[Scikit-Learn Models](sklearn_models.md)** |
| **PyTorch** | Neural network on molecular descriptors | **[PyTorch Models](pytorch_models.md)** |
| **Fingerprint** | Count fingerprint models for molecular similarity | **[Fingerprint Models](fingerprint_models.md)** |
| **ChemProp** | Message Passing Neural Network on molecular graphs | **[ChemProp Models](chemprop_models.md)** |
| **Meta Model** | Ensemble aggregating multiple endpoints | **[Meta Models](meta_models.md)** |

## Quick Example

```python
from workbench.api import FeatureSet, ModelType, ModelFramework

fs = FeatureSet("my_features")

# XGBoost (default framework)
xgb_model = fs.to_model(
    name="my-xgb-model",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.XGBOOST,
    target_column="target",
    feature_list=fs.feature_columns,
)

# PyTorch (same API, different framework)
pytorch_model = fs.to_model(
    name="my-pytorch-model",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.PYTORCH,
    target_column="target",
    feature_list=fs.feature_columns,
)

# Deploy Endpoints
xgb_end = xgb_model.to_endpoint()
xgb_end.auto_inference()

pytorch_end = pytorch_model.to_endpoint()
pytorch_end.auto_inference()
```

---

## Questions?

<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)
