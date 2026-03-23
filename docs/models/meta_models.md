# Meta Models (Ensembles)

!!! tip inline end "Ensemble Benefits"
    Meta models combine predictions from multiple endpoints (XGBoost, PyTorch, ChemProp, etc.) to improve accuracy and robustness. Different model frameworks often make different errors, so aggregating them can reduce overall prediction error.

Meta models aggregate predictions from multiple deployed endpoints into a single ensemble prediction. Rather than training on raw features, a meta model calls existing endpoints at inference time and combines their outputs using confidence-weighted voting strategies.

## Why Use a Meta Model?

Different model frameworks capture different aspects of molecular structure:

- **XGBoost** excels at tabular features (RDKit descriptors)
- **PyTorch** can learn nonlinear descriptor interactions
- **ChemProp** operates directly on molecular graphs

By combining predictions across frameworks, you get:

- **Lower error** — individual model mistakes get averaged out
- **Better calibration** — ensemble disagreement provides a natural uncertainty signal
- **Robustness** — no single model failure dominates the prediction

## Quick Start

### Simulate Ensemble Performance

Before creating a meta model, you can simulate how different aggregation strategies would perform using existing endpoint predictions:

```python
from workbench.api import MetaModel

# Simulate ensemble performance across endpoints
sim = MetaModel.simulate(["logd-xgb-end", "logd-pytorch-end", "logd-chemprop-end"])
sim.report()
```

**Example Output**

```
===  Individual Model Performance  ===
logd-xgb-end:      MAE=0.428  RMSE=0.580  R²=0.817
logd-pytorch-end:   MAE=0.445  RMSE=0.594  R²=0.808
logd-chemprop-end:  MAE=0.412  RMSE=0.558  R²=0.831

===  Ensemble Strategy Comparison  ===
Strategy                     MAE     RMSE    R²
simple_mean                 0.391   0.532   0.846
confidence_weighted         0.388   0.528   0.849
inverse_mae_weighted        0.385   0.525   0.850
scaled_conf_weighted        0.383   0.522   0.852  ← Best
calibrated_conf_weighted    0.384   0.523   0.851
drop_worst                  0.398   0.541   0.841
```

The simulator analyzes all aggregation strategies and identifies the one that gives the best performance on held-out cross-fold data.

### Create a Meta Model

Once you're satisfied with the simulation results, create the meta model. It auto-simulates internally to pick the best strategy:

```python
from workbench.api import MetaModel

# Create a meta model (auto-simulates to find best strategy)
meta = MetaModel.create(
    name="logd-meta",
    endpoints=["logd-xgb-end", "logd-pytorch-end", "logd-chemprop-end"],
    description="Meta model for LogD prediction",
    tags=["meta", "logd", "ensemble"],
)
print(meta.summary())
```

### Deploy the Meta Model

Meta models deploy like any other Workbench model:

```python
# Deploy to an AWS Endpoint
endpoint = meta.to_endpoint(tags=["meta", "logd"])

# Run inference — the meta endpoint calls child endpoints internally
results_df = endpoint.inference(my_dataframe)
```

At inference time, the meta endpoint calls all child endpoints in parallel, collects their predictions and confidence scores, and aggregates them using the selected strategy.

## Aggregation Strategies

The meta model supports five aggregation strategies for combining endpoint predictions:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Strategy</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Description</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">When to Use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">simple_mean</td>
      <td style="padding: 8px 16px;">Equal weight to all endpoints</td>
      <td style="padding: 8px 16px;">Baseline; all models perform similarly</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">confidence_weighted</td>
      <td style="padding: 8px 16px;">Weight by per-row confidence score</td>
      <td style="padding: 8px 16px;">Models have well-calibrated confidence</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">inverse_mae_weighted</td>
      <td style="padding: 8px 16px;">Static weights from inverse MAE (lower error = higher weight)</td>
      <td style="padding: 8px 16px;">Default; good when per-row confidence isn't reliable</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">scaled_conf_weighted</td>
      <td style="padding: 8px 16px;">Inverse-MAE weights × per-row confidence</td>
      <td style="padding: 8px 16px;">Best of both worlds; often the top performer</td>
    </tr>
    <tr>
      <td style="padding: 8px 16px; font-weight: bold;">calibrated_conf_weighted</td>
      <td style="padding: 8px 16px;">Confidence scaled by |confidence–error correlation|</td>
      <td style="padding: 8px 16px;">Rewards models whose confidence actually predicts accuracy</td>
    </tr>
  </tbody>
</table>

### How Strategies Work

All strategies produce **per-row weights** that are normalized to sum to 1. The final prediction is:

```
prediction = Σ (weight_i × prediction_i)
```

For the **confidence-based** strategies, each row can have different weights — if one model is very confident about a particular compound and another is not, the confident model gets more influence on that row.

### Zero-Confidence Fallback

When all child endpoints report zero confidence for a given row (e.g., the compound is outside the training domain for all models), the confidence-weighted strategies fall back to static inverse-MAE weights. This prevents degenerate predictions and ensures every row gets a reasonable ensemble output.

## Simulation Deep Dive

The simulator provides several methods for detailed analysis:

```python
sim = MetaModel.simulate(["logd-xgb-end", "logd-pytorch-end", "logd-chemprop-end"])

# Full report with all strategies
sim.report()

# Get the best strategy configuration (used internally by create())
config = sim.get_best_strategy_config()
print(config)
# {'aggregation_strategy': 'scaled_conf_weighted',
#  'model_weights': {'logd-xgb-end': 0.34, 'logd-pytorch-end': 0.30, ...},
#  'corr_scale': {'logd-xgb-end': 0.62, 'logd-pytorch-end': 0.55, ...},
#  'endpoints': ['logd-xgb-end', 'logd-pytorch-end', 'logd-chemprop-end'],
#  'target_column': 'logd'}

# Export the best ensemble's predictions to CSV
df = sim.best_ensemble_predictions()
df.to_csv("ensemble_predictions.csv", index=False)
```

### Drop-Worst Analysis

The simulator also evaluates whether removing the worst-performing model improves the ensemble. If dropping a model reduces error, the `get_best_strategy_config()` method returns the reduced endpoint list. This is handled automatically by `MetaModel.create()`.

## CLI Tool

The `meta_model_sim` CLI provides quick ensemble analysis from the command line:

```bash
# Simulate ensemble performance
meta_model_sim logd-xgb-end logd-pytorch-end logd-chemprop-end

# Use a specific inference capture
meta_model_sim logd-xgb-end logd-pytorch-end logd-chemprop-end \
    --capture-name full_cross_fold

# Save best ensemble predictions to CSV
meta_model_sim logd-xgb-end logd-pytorch-end logd-chemprop-end \
    --output ensemble_results.csv
```

## How It Works Under the Hood

### Creation Flow

When you call `MetaModel.create()`, the following happens:

1. **Lineage resolution** — Backtraces the first endpoint's lineage (endpoint → model → FeatureSet) to automatically resolve the target column, ID column, and feature list
2. **Simulation** — Runs `MetaModelSimulator` to evaluate all aggregation strategies on cross-fold prediction data
3. **Strategy selection** — Picks the best-performing strategy, including checking if dropping the worst model helps
4. **Training job** — Runs a minimal SageMaker training job that saves the meta configuration (endpoints, weights, strategy) as a model artifact
5. **Registration** — Creates a SageMaker Model Package with the meta inference container
6. **Metadata** — Sets Workbench metadata (model type, framework, features, endpoints)

### Inference Flow

When a meta endpoint receives a prediction request:

1. **Parse input** — Reads the incoming CSV/JSON data
2. **Fan out** — Calls all child endpoints in parallel using `fast_inference`
3. **Aggregate** — Combines predictions using the stored aggregation strategy and weights
4. **Return** — Outputs aggregated prediction, prediction_std (ensemble disagreement), and confidence

## API Reference

::: workbench.api.meta_model

---

## Questions?

<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)
