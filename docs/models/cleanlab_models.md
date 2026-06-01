# Cleanlab Models

Workbench integrates [cleanlab](https://docs.cleanlab.ai) for **data-quality and label-quality analysis** — finding mislabeled or noisy samples in your training data and estimating per-sample model uncertainty. It works for both regression and classification.

Workbench handles the data prep (numeric-feature filtering, dropping NaNs, label encoding) and joins cleanlab's results back to your **ID column**, so flagged samples are traceable to real records. The underlying cleanlab objects are exposed directly for anything beyond the curated helpers.

## Optional Dependency

Cleanlab is **not** installed by `workbench` or any of its extras. Install it separately before use:

```bash
pip install 'cleanlab[datalab]>=2.8.0'
```

!!! note "Why 2.8.0+"
    Cleanlab 2.8.0+ resolves an earlier Datalab incompatibility with `datasets` 4.x ([cleanlab#1253](https://github.com/cleanlab/cleanlab/issues/1253)).

## Quick Start

The easiest entry point is from an existing `Model` — it infers the target, features, and ID column from the model's metadata:

```python
from workbench.api import Model

model = Model("aqsol-regression")
cleanlab = model.cleanlab_model()

# Samples most likely to be mislabeled (worst quality first, keyed by ID)
issues = cleanlab.label_issues()
print(issues.head(10))
```

You can also start from a `FeatureSet`, specifying the target and features explicitly (a FeatureSet has no notion of which column is the target):

```python
from workbench.api import FeatureSet, ModelType

fs = FeatureSet("aqsol_features")
cleanlab = fs.cleanlab_model(
    target="solubility",
    features=["mollogp", "molwt", "numhdonors"],
    model_type=ModelType.REGRESSOR,
)
issues = cleanlab.label_issues()
```

Both entry points return a `CleanlabModels` instance.

## Workbench Helpers

These are the recommended surface for most uses. Per-sample results come back as DataFrames keyed by your ID column.

| Method | Returns | Notes |
|---|---|---|
| `label_issues()` | DataFrame | One row per sample, ID column first, sorted by `label_quality` (worst first). Includes `is_label_issue`, `given_label`, `predicted_label`. Regression + classification. |
| `epistemic_uncertainty()` | DataFrame | `[id_column, epistemic_uncertainty]`, sorted descending. Reducible (model) uncertainty. **Regression only.** |
| `aleatoric_uncertainty()` | float | Dataset-level irreducible noise estimate. **Regression only.** |

```python
# Label issues — worst-quality samples first, traceable by ID
issues = cleanlab.label_issues()
flagged = issues[issues["is_label_issue"]]
print(f"{len(flagged)} potential label issues out of {len(issues)} samples")

# Epistemic uncertainty — samples the model is least sure about (regression)
uncertain = cleanlab.epistemic_uncertainty()
print(uncertain.head(10))

# Aleatoric uncertainty — irreducible data noise (regression, single value)
print(f"Dataset noise estimate: {cleanlab.aleatoric_uncertainty():.4f}")
```

## Working with Native Cleanlab Objects

For anything beyond the helpers, the underlying cleanlab objects are exposed **unmodified** — use them with cleanlab's own API (see the [cleanlab docs](https://docs.cleanlab.ai)):

```python
# Native cleanlab CleanLearning model (fitted)
cl = cleanlab.clean_learning()
predictions = cl.predict(X)

# Native cleanlab Datalab — comprehensive data-quality report
lab = cleanlab.datalab()
lab.report()
issue_summary = lab.get_issue_summary()
```

!!! note "Division of labor"
    Cleanlab owns the modeling API (exposed via `clean_learning()` / `datalab()`); Workbench owns data prep and joining results to your ID column (the helper methods above).

## Regression vs. Classification

- **Both** support `label_issues()` and the native `clean_learning()` / `datalab()` objects.
- **Regression only**: `epistemic_uncertainty()` and `aleatoric_uncertainty()`. Calling these on a classification model raises a clear `TypeError`.

---

## Questions?

<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)
