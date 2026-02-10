# Confidence Scores in Workbench
!!!tip inline end "Need Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and Workbench. So please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)

Workbench provides **confidence scores** for every model prediction, giving users a measure of how much to trust each prediction. Higher confidence means the ensemble models agree closely; lower confidence means they disagree.

## Overview

Every Workbench model — XGBoost, PyTorch, or ChemProp — is a **5-model ensemble** trained via cross-validation. The same uncertainty quantification pipeline runs for all three frameworks:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Framework</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Ensemble</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Std Source</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Calibration</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding: 8px 16px; color: #ff9f43; font-weight: bold;">XGBoost</td><td style="padding: 8px 16px;">5-fold CV, XGBRegressor per fold</td><td style="padding: 8px 16px;"><code>np.std</code> across 5 predictions</td><td style="padding: 8px 16px;">Conformal scaling</td></tr>
    <tr><td style="padding: 8px 16px; color: #3a86ff; font-weight: bold;">PyTorch</td><td style="padding: 8px 16px;">5-fold CV, TabularMLP per fold</td><td style="padding: 8px 16px;"><code>np.std</code> across 5 predictions</td><td style="padding: 8px 16px;">Conformal scaling</td></tr>
    <tr><td style="padding: 8px 16px; color: #00d4aa; font-weight: bold;">ChemProp</td><td style="padding: 8px 16px;">5-fold CV, MPNN per fold</td><td style="padding: 8px 16px;"><code>np.std</code> across 5 predictions</td><td style="padding: 8px 16px;">Conformal scaling</td></tr>
  </tbody>
</table>

## Three-Step Pipeline

### 1. Ensemble Disagreement

Each fold of the 5-fold cross-validation produces a model trained on a different slice of the data. At inference time, all 5 models make a prediction and we take the average. The **standard deviation** across the 5 predictions (`prediction_std`) is the raw uncertainty signal.

When the models agree closely (low std), the prediction is more reliable. When they disagree (high std), something about that compound is tricky.

### 2. Conformal Calibration

Raw ensemble std tells you *which* predictions to trust more, but the numbers aren't calibrated — a std of 0.3 doesn't map to a meaningful interval. Workbench uses **conformal prediction** to fix this:

1. Compute nonconformity scores on held-out validation data: `score = |actual - predicted| / std`
2. For each confidence level (50%, 68%, 80%, 90%, 95%), find the quantile of scores that achieves the target coverage
3. Build intervals: `prediction ± scale_factor × std`

The scaling factors are computed once during training and stored as metadata. At inference, building intervals is a simple multiply.

The result: prediction intervals that vary per-compound (based on ensemble disagreement) but are calibrated to achieve correct coverage. An 80% interval really does contain ~80% of true values.

### 3. Percentile-Rank Confidence

Confidence is the **percentile rank** of each prediction's `prediction_std` within the training set's std distribution:

```
confidence = 1 - percentile_rank(prediction_std)
```

- **Confidence 0.7** means this prediction's ensemble disagreement is lower than 70% of the training set — a relatively tight prediction.
- **Confidence 0.1** means 90% of training predictions had lower uncertainty — this compound is an outlier.

This approach gives scores that spread across the full 0–1 range, are directly interpretable, and require no arbitrary parameters.

## Interpreting Confidence Scores

### High Confidence (> 0.7)
- Ensemble models agree closely on the prediction
- Prediction intervals are narrower than most training predictions
- Good candidates for prioritization

### Medium Confidence (0.3 – 0.7)
- Typical level of ensemble disagreement
- Predictions are likely reasonable but verify important decisions

### Low Confidence (< 0.3)
- Ensemble models disagree significantly
- Prediction intervals are wider than most training predictions
- May indicate out-of-distribution compounds or regions where the model is uncertain

## What Confidence Doesn't Tell You

Confidence reflects how much the ensemble models agree — but agreement doesn't guarantee correctness:

- **High confidence ≠ correct prediction.** It means the models agree, not that they're right.
- **Novel chemistry may get falsely high confidence** if it happens to fall in a region where models extrapolate consistently.
- **Confidence is relative to the training set.** A confidence of 0.9 from a kinase solubility model doesn't transfer to a PROTAC dataset.

For truly out-of-distribution detection, consider pairing confidence with applicability domain analysis.

## Metrics for Evaluating Confidence

Workbench computes several metrics to evaluate how well confidence correlates with actual prediction quality:

### confidence_to_error_corr
Spearman correlation between confidence and absolute error. **Should be negative** (high confidence = low error). Target: < -0.5

### interval_to_error_corr
Spearman correlation between interval width and absolute error. **Should be positive** (wide intervals = high error). Target: > 0.5

### Coverage Metrics
For each confidence level (50%, 68%, 80%, 90%, 95%), the percentage of true values that fall within the prediction interval. Should match the target coverage.

## Deep Dive

For more details on the approach, including code walkthrough and validation results, see the [Model Confidence Blog](../blogs/model_confidence.md).

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

Need help with confidence scores or uncertainty quantification? Want to develop a customized application tailored to your business needs?

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
