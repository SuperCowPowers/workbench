# Advanced Models

!!! tip inline end "OpenADMET Challenge"
    ChemProp was used by many of the top performers on the [OpenADMET Leaderboard](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge). Workbench makes it easy to train and deploy ChemProp models to AWS®.

Workbench supports advanced model frameworks including PyTorch neural networks and ChemProp message passing neural networks (MPNNs). These models can be trained and deployed to AWS® with the same simple API as other Workbench models.

## Available Model Frameworks

| Model Framework | Description |
|-----------------|-------------|
| **XGBoost** | Gradient boosted trees on RDKit molecular descriptors |
| **PyTorch** | Neural network on RDKit molecular descriptors |
| **ChemProp** | Message Passing Neural Network (MPNN) on molecular graphs |
| **ChemProp Hybrid** | MPNN combined with top RDKit descriptors |
| **ChemProp Multi-Task** | Single MPNN predicting multiple endpoints simultaneously |

## Why ChemProp?

Traditional models treat molecules as a list of computed descriptors. ChemProp takes a different approach—it operates directly on the molecular graph structure, using atoms as nodes and bonds as edges. This allows the model to learn representations from the molecular topology itself.

In the OpenADMET Challenge, ChemProp models consistently performed well across the ADMET endpoints given for the contest:

- LogD (Lipophilicity)
- KSOL (Kinetic Solubility)
- HLM/MLM CLint (Liver Clearance)
- Caco-2 Permeability & Efflux
- Plasma & Brain Protein Binding

## Deploying a ChemProp Model

Creating and deploying a ChemProp model follows the standard Workbench pattern:

```python
from workbench.api import DataSource, FeatureSet, ModelType, ModelFramework

# Create a DataSource and FeatureSet
ds = DataSource("my_molecules.csv", name="admet_data")
fs = ds.to_features("admet_features", id_column="mol_id")

# Create a ChemProp model
model = FeatureSet("admet_features").to_model(
    name="my-chemprop-model",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    target_column="logd",
    feature_list=["smiles"],
    description="ChemProp model for LogD prediction",
)

# Deploy to an AWS Endpoint
endpoint = model.to_endpoint()
```

## Multi-Task Models

ChemProp supports multi-task learning, where a single model predicts multiple endpoints simultaneously. This can improve performance when endpoints are related and share underlying molecular features.

```python
# Define multiple target columns for multi-task learning
ADMET_TARGETS = [
    'logd', 'ksol', 'hlm_clint', 'mlm_clint',
    'caco_2_papp_a_b', 'caco_2_efflux',
    'mppb', 'mbpb', 'mgmb'
]

model = feature_set.to_model(
    name="admet-multi-task",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    target_column=ADMET_TARGETS,  # List enables multi-task
    feature_list=["smiles"],
    tags=["chemprop", "multitask"],
)
```

## Confidence Estimates

All Workbench models include built-in uncertainty quantification. This provides confidence estimates alongside predictions, which is valuable for drug discovery workflows:

- **High confidence**: Predictions can be trusted for decision-making
- **Low confidence**: The molecule may be outside the training domain; consider gathering additional data

<figure>
  <img src="images/confidence.jpg" alt="Confidence Estimates">
  <figcaption>Confidence Models: All our models provide confidence metrics to identify predictions where the model is unsure or needs more data.</figcaption>
</figure>

## PyTorch Models

For some assays, PyTorch models on RDKit descriptors can outperform ChemProp. These models train faster and work well when molecular descriptors capture the relevant features:

```python
model = feature_set.to_model(
    name="my-pytorch-model",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.PYTORCH,
    target_column="logd",
    feature_list=fs.feature_columns,
)
```

!!! note "Examples"
    All Workbench model examples are in the repository under the `examples/models` directory. For full code listings, visit [Workbench Model Examples](https://github.com/SuperCowPowers/workbench/blob/main/examples/models).

---

!!! warning "Beta Software"
    Workbench is currently in beta. We're actively looking for beta testers! If you're interested in early access, contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com).

## Questions?

<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)

## References

- **ChemProp:** Yang et al. "Analyzing Learned Molecular Representations for Property Prediction" *J. Chem. Inf. Model.* 2019 — [GitHub](https://github.com/chemprop/chemprop) | [Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237)
- **PyTorch:** Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library" *NeurIPS* 2019 — [pytorch.org](https://pytorch.org/) | [Paper](https://arxiv.org/abs/1912.01703)
- **OpenADMET Challenge:** Community benchmark for ADMET property prediction — [Leaderboard](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge) | [GitHub](https://github.com/OpenADMET/OpenADMET-ExpansionRx-Challenge)

<i>® Amazon Web Services, AWS, the Powered by AWS logo, are trademarks of Amazon.com, Inc. or its affiliates</i>
