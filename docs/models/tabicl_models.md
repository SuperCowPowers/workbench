# TabICL Models

[TabICL](https://github.com/soda-inria/tabicl) is a tabular foundation model: a transformer pretrained on synthetic tables. Workbench supports it as a regression framework, with the same training, deployment, inference, and confidence scoring as every other framework.

TabICL learns *in context*. Nothing is trained on your data; the model reads your training rows at prediction time and predicts from them in one forward pass. That makes it quick to build and strong on small and medium datasets, and it changes how the model is served (see [Endpoints](#endpoints-serverless-or-real-time)).

## Creating a TabICL Model

```python
from workbench.api import FeatureSet, ModelType, ModelFramework

fs = FeatureSet("aqsol_features")
features = ["molwt", "mollogp", "molmr", "heavyatomcount", "numhacceptors", "numhdonors", "tpsa"]

# Regression with uncertainty quantification
model = fs.to_model(
    name="sol-tabicl-reg",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.TABICL,
    target_column="solubility",
    feature_list=features,
    description="TabICL regression for solubility",
    tags=["tabicl", "solubility"],
)

# Deploy and run inference
endpoint = model.to_endpoint()
endpoint.test_inference()
```

TabICL reads tabular features, so give it descriptor columns (RDKit, Mordred, or your own), not SMILES. Missing values and categorical columns are handled by TabICL itself.

## How a TabICL Model Is Built

| Step | What happens |
|------|--------------|
| **Cross-validation** | Fold models produce out-of-fold predictions for the model's metrics and uncertainty calibration. They are not served. |
| **Served model** | One fit on every training row. The rows are cached inside the model, so a request costs only its own rows. |
| **Uncertainty** | TabICL's own predictive distribution feeds Workbench's confidence model. |
| **Feature importance** | SHAP values from TabICL's explainer. |
| **Memory probe** | The training job measures how much memory the saved model needs to serve. |

A fold model sees 80% of the rows as context, so the cross-fold metrics are a slightly conservative estimate of the served model, which sees all of them.

## Hyperparameters

```python
model = fs.to_model(
    name="sol-tabicl-pca",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.TABICL,
    target_column="solubility",
    feature_list=features,
    hyperparameters={"pca_components": 4},
)
```

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `n_folds` | `5` | Cross-validation folds for out-of-fold metrics (1 = a single train/validation split) |
| `n_estimators` | `8` | TabICL's internal ensemble: members see different column orders and scalings |
| `batch_size` | `1` | Ensemble members per forward pass. 1 gives the lowest peak memory; predictions are the same at any value |
| `pca_components` | `None` | Standardize the features and reduce them with PCA before TabICL: an integer is a component count, a fraction between 0 and 1 is an explained-variance target (`0.95` keeps the fewest components explaining 95%) |
| `top_variance_features` | `None` | Keep this many of the columns expanded from compressed features (highest variance on the training rows); other features are kept |
| `shap_sample_size` | `100` | Rows explained for SHAP feature importance (0 disables) |
| `split_strategy` | `"scaffold"` | `"scaffold"`, `"butina"`, or `"random"` (scaffold and butina need a `smiles` column) |
| `butina_cutoff` | `0.4` | Tanimoto distance cutoff for Butina clustering |
| `seed` | `42` | Random seed |
| `uq_version` | `"v1"` | Confidence model: `"v0"`, `"v1"`, or `"v2"` |

### Feature Count

TabICL is pretrained on tables of up to 100 columns. More features work, and the training log notes when a model is past that range. If accuracy suffers, two hyperparameters reduce them; set one, not both.

- `top_variance_features` is for a compressed feature such as an embedding or a fingerprint, which expands into hundreds of columns. It keeps the expanded columns with the highest variance on the training rows and leaves every other feature in place, so extra descriptors next to an embedding are untouched:

    ```python
    fs.set_compressed_features(["monroe"])
    model = fs.to_model(
        ...,
        model_framework=ModelFramework.TABICL,
        feature_list=["monroe", "logp"],
        hyperparameters={"top_variance_features": 100},  # 720 embedding columns -> 100
    )
    ```

    In our tests, 100 columns did as well as 360 or all 720, within test-set noise, and needs the least serving memory.

- `pca_components` standardizes every feature and projects them all onto this many components. PCA needs numeric features, so it can't be combined with categorical columns.

The [Monroe Embeddings + Tabular Foundation Models](../blogs/monroe_tabicl.md) blog walks through the Monroe embedding endpoint and pairing it with TabICL.

## Endpoints: Serverless or Real-Time

A TabICL model carries its training rows, so its memory grows with the training set. The training job measures the memory the model needs to serve, and `to_endpoint()` uses that measurement.

**Serverless (the default)** works for smaller training sets. When the measured memory is over what a serverless endpoint can give a model, `to_endpoint()` raises and tells you so, rather than deploying an endpoint that fails to load:

```
ValueError: sol-tabicl-reg needs 9.17 GB to serve, over the 5.5 GB a serverless
endpoint can give a model. Deploy a real-time endpoint with to_endpoint(serverless=False);
its instance is sized from the measured memory.
```

**Real-time** endpoints size their instance from the same measurement:

```python
endpoint = model.to_endpoint(serverless=False)
```

A real-time instance runs, and bills, around the clock, which is why the switch is yours to make. Pass `instance="ml.r7i.xlarge"` (or any instance type) to choose one yourself.

As a guide, with 17 features:

| Training rows | Measured memory | Endpoint | 100-row request |
|---------------|-----------------|----------|-----------------|
| 5,000 | 5.25 GB | Serverless | about 9 seconds |
| 10,000 | 9.17 GB | Real-time, `ml.r7i.large` | about 4 seconds |

The measured memory is the peak while the model loads, which is well above what it settles to afterwards. An endpoint has to survive that peak.

## Confidence and Uncertainty

TabICL predicts a full distribution for each row. Workbench takes the width of that distribution as the model's `prediction_std` and feeds it to the same confidence model the other frameworks use, so a TabICL endpoint returns the same columns: `prediction`, `prediction_std`, `confidence`, and the calibrated intervals (`q_025` … `q_975`).

See [Model Confidence](../confidence/index.md) for how the confidence model works.

## Feature Importance

SHAP values come from TabICL's own explainer and are stored with the model like any other framework's. The explainer works on raw numeric features, so SHAP is skipped when the model uses `pca_components` or `top_variance_features`, or has categorical columns.

## Limits

- **Regression only.** `ModelType.REGRESSOR` and `ModelType.UQ_REGRESSOR`.
- **No sample weights.** TabICL has no per-row weights; use `validation_ids` to hold rows out of training.
- **Memory grows with training rows**, as described above.

## Pretrained Weights

Training resolves the TabICL checkpoint the same way as other foundation weights, with no setup:

1. **Local cache** — `~/.workbench/foundation/`
2. **Account mirror** (optional) — `s3://$WORKBENCH_BUCKET/foundation-models/tabicl/...`
3. **Public bucket** — `s3://workbench-public-data/foundation-models/tabicl/...`, read anonymously

A mirror is only for an account whose training jobs can't reach public S3; see [Foundation Weight Storage](chemprop_models.md#foundation-weight-storage). The served model includes the weights, so an endpoint never downloads anything.

## Running Locally

TabICL is part of the `modeling` extra, so a local model needs no AWS account and no SageMaker images:

```bash
pip install "workbench[modeling]"
```

!!! note "Examples"
    Full code listings: [`examples/models/tabicl_model.py`](https://github.com/SuperCowPowers/workbench/blob/main/examples/models/tabicl_model.py) (AWS) and [`examples/models/tabicl_local.py`](https://github.com/SuperCowPowers/workbench/blob/main/examples/models/tabicl_local.py) (local).

---

## Questions?

<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)

## References

- **TabICL** — [github.com/soda-inria/tabicl](https://github.com/soda-inria/tabicl) (BSD-3-Clause, code and weights); [arXiv:2502.05564](https://arxiv.org/abs/2502.05564), [arXiv:2602.11139](https://arxiv.org/abs/2602.11139)
