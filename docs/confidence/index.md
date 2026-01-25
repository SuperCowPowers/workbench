# Confidence Scores in Workbench
!!!tip inline end "Need Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and Workbench. So please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)

Workbench provides **confidence scores** for model predictions, giving users a measure of how certain the model is about each prediction. Higher confidence indicates the model is more certain, while lower confidence suggests greater uncertainty.

## Overview

| **Framework**  | **UQ Method**                          | **Confidence Source**           |
|----------------|----------------------------------------|---------------------------------|
| XGBoost        | MAPIE Conformalized Quantile Regression | 80% prediction interval width  |
| PyTorch        | MAPIE Conformalized Quantile Regression | 80% prediction interval width  |
| ChemProp       | Ensemble disagreement (5 models)       | Ensemble standard deviation     |

All frameworks use the same confidence formula, but the uncertainty source differs.

## Confidence Formula

Confidence is computed using a simple exponential decay function:

```
confidence = exp(-uncertainty / median_uncertainty)
```

Where:

- **uncertainty**: The model's uncertainty estimate for each prediction
- **median_uncertainty**: The median uncertainty from the training/validation data


## XGBoost and PyTorch: MAPIE CQR

For XGBoost and PyTorch models, Workbench uses **MAPIE Conformalized Quantile Regression (CQR)** to generate prediction intervals with coverage guarantees.

### How It Works

1. **Train quantile models**: LightGBM models are trained to predict the 10th and 90th percentiles (80% confidence interval)
2. **Conformalize**: MAPIE adjusts the intervals using a held-out calibration set to guarantee coverage
3. **Compute interval width**: The uncertainty is the width of the 80% prediction interval (q_90 - q_10)
4. **Calculate confidence**: Apply the exponential decay formula

### Why 80% Confidence Interval?

We use the 80% CI (q_10 to q_90) rather than other intervals because:

- **68% CI** is too narrow and sensitive to noise
- **95% CI** is too wide and less discriminating between samples
- **80% CI** provides a good balance for ranking prediction reliability

### Coverage Guarantees

MAPIE's conformalization ensures that prediction intervals achieve their target coverage. For example, an 80% CI will contain approximately 80% of true values on the calibration set. This is a key advantage over simple quantile regression.

## ChemProp: Ensemble Disagreement

For ChemProp models, Workbench trains an **ensemble of 5 models** and uses their disagreement as the uncertainty measure.

### How It Works

1. **Train ensemble**: 5 ChemProp models are trained with different random seeds
2. **Predict**: Each model makes a prediction for each sample
3. **Compute disagreement**: The standard deviation across the 5 predictions is the uncertainty
4. **Calibrate**: The standard deviation is empirically calibrated against actual errors
5. **Calculate confidence**: Apply the exponential decay formula using calibrated std

### Ensemble vs MAPIE

| Aspect | MAPIE (XGBoost/PyTorch)   | Ensemble (ChemProp)        |
|--------|---------------------------|----------------------------|
| Coverage guarantee | Yes (conformal)           | No (empirical)             |
| Computational cost | + 5 MAPIE models          | Just the 5 chemprop models |
| Uncertainty type | Prediction interval width | Model disagreement         |

Both approaches are valid and widely used in the ML community. MAPIE provides theoretical guarantees, while ensembles capture model uncertainty more directly.

## Interpreting Confidence Scores

### High Confidence (> 0.5)
- Model is relatively certain about the prediction
- Prediction interval is narrower than typical
- Good candidates for prioritization

### Medium Confidence (0.3 - 0.5)
- Model has typical uncertainty
- Prediction is likely reasonable but verify important decisions

### Low Confidence (< 0.3)
- Model is uncertain about the prediction
- Prediction interval is wider than typical
- May indicate out-of-distribution samples or difficult predictions

## Metrics for Evaluating Confidence

Workbench computes several metrics to evaluate how well confidence correlates with actual prediction quality:

### confidence_to_error_corr
Spearman correlation between confidence and absolute error. **Should be negative** (high confidence = low error). Target: < -0.5

### interval_to_error_corr
Spearman correlation between interval width and absolute error. **Should be positive** (wide intervals = high error). Target: > 0.5

### Coverage Metrics
For each confidence level (50%, 68%, 80%, 90%, 95%), the percentage of true values that fall within the prediction interval. Should match the target coverage.

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

Need help with confidence scores or uncertainty quantification? Want to develop a customized application tailored to your business needs?

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
