# Advanced Models

!!! tip inline end "OpenADMET Challenge"
    ChemProp was used by many of the top performers on the [OpenADMET Leaderboard](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge). Workbench makes it easy to train and deploy these models to AWS®.

Workbench supports advanced model frameworks beyond the default XGBoost. All frameworks use the same API — just change the `model_framework` parameter.

## Available Model Frameworks

| Model Framework | Description | Details |
|-----------------|-------------|---------|
| **XGBoost** | Gradient boosted trees on RDKit molecular descriptors | Default framework |
| **PyTorch** | Neural network on RDKit molecular descriptors | Good for nonlinear descriptor interactions |
| **[ChemProp](chemprop_models.md)** | Message Passing Neural Network (MPNN) on molecular graphs | Learns directly from molecular topology |
| **[ChemProp Hybrid](chemprop_models.md#hybrid-chemprop)** | MPNN combined with top RDKit descriptors | Best of both worlds |
| **[ChemProp Multi-Task](chemprop_models.md#multi-task-chemprop)** | Single MPNN predicting multiple endpoints | Related endpoints with shared chemistry |
| **[ChemProp Foundation](chemprop_models.md#foundation-chemprop)** | Pretrained MPNN weights (CheMeleon) | Transfer learning for small datasets |
| **[Meta Model](../meta_models/index.md)** | Ensemble aggregating multiple endpoints | Combines frameworks for lower error |

## Quick Example

```python
from workbench.api import FeatureSet, ModelType, ModelFramework

fs = FeatureSet("admet_features")

# ChemProp model
chemprop_model = fs.to_model(
    name="logd-chemprop",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    target_column="logd",
    feature_list=["smiles"],
)

# PyTorch model (same API, different framework)
pytorch_model = fs.to_model(
    name="logd-pytorch",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.PYTORCH,
    target_column="logd",
    feature_list=fs.feature_columns,
)

# Deploy either one
endpoint = chemprop_model.to_endpoint()
```

For detailed documentation on ChemProp model types (ST, MT, Hybrid, Foundation), see **[ChemProp Models](chemprop_models.md)**. For ensemble models that combine multiple endpoints, see **[Meta Models](../meta_models/index.md)**.

---

!!! warning "Beta Software"
    Workbench is currently in beta. We're actively looking for beta testers! If you're interested in early access, contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com).

## Questions?

<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)

<i>® Amazon Web Services, AWS, the Powered by AWS logo, are trademarks of Amazon.com, Inc. or its affiliates</i>
