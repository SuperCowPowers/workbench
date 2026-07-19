# Uncertainty Quantification

A prediction without a confidence is half an answer. In drug discovery, knowing
*how much to trust* a number often matters more than the number — it decides
whether someone synthesizes a compound.

## The foundation: ensemble disagreement

Every Workbench model — XGBoost, PyTorch, ChemProp alike — is a **5-model
ensemble trained by cross-validation**. All 5 predict, the mean is the
prediction, and their spread is `prediction_std`. When the members disagree, the
prediction is less reliable.

`prediction_std` is soft log-compressed above the IQR fence. That is monotonic
(rankings and intervals are unaffected), but it means you should read it as an
**uncertainty score, not a literal standard deviation**.

Raw std ranks well but isn't calibrated — it tells you *which* predictions to
trust, not that ±0.3 means anything. The UQ versions are what turn it into
calibrated confidence and intervals.

## Making a UQ model

```python
model = fs.to_model(
    name="my-model-reg",
    model_type=ModelType.UQ_REGRESSOR,
    target_column="solubility",
    feature_list=features,
    hyperparameters={"uq_version": "v1"},
)
```

Always set `uq_version: "v1"` on new models. The code default is **`v0`**, so
leaving it unset silently gives the weaker calibrator.

## What comes back

```
prediction, prediction_std, expected_residual, confidence,
q_025, q_05, q_25, q_50, q_75, q_95, q_975
```

`q_05`/`q_95` bound the 90% interval. `confidence` is a percentile rank: 0.7
means "expected error lower than 70% of calibration-set predictions."

`inference()` and `cross_fold_inference()` take `include_quantiles=False` by
default — pass `True` to persist the `q_*` columns.

## The three regression versions

All three are fit at training time and saved in the bundle; `uq_version` picks
the active one.

| | Status | Approach | SMILES? |
|---|---|---|---|
| **v0** | beta, code default | Binned isotonic `(std -> \|residual\|)` + split conformal | No |
| **v1** | beta, **recommended** | RandomForest error model on neighborhood features + normalized conformal | Yes |
| **v2** | experimental | Pure applicability-domain score from fingerprint proximity | Yes |

**v1 and v2 require a SMILES column** to build the fingerprint proximity set.
Without one, only v0 is fit and used — so a no-SMILES model silently gets v0
regardless of what you asked for.

**The failure v1 fixes:** the ensemble can agree unanimously and still be wrong,
typically near censoring boundaries. Kinetic solubility assays cap around
-3.5 LogS, so a big training cluster sits there; a similar compound whose true
value is -5.5 gets all 5 members converging on -3.6. Agreement is genuine but
uninformative — *confidently wrong*, and raw std cannot see it. v1's
`knn_target_std` (do the neighbors even agree on the label?) and
`local_pred_gap` catch exactly this. Its five features are
`[prediction, prediction_std, knn_distance, knn_target_std, local_pred_gap]`.

**v2's distinctive bit:** intervals come from the neighbors' target values
centered on the neighbor *median*, not the prediction. So when the model's
marker falls outside the interval, that gap is itself a **cliff diagnostic** —
the model is extrapolating past its local support.

Compare versions offline without redeploying:

```python
uq_v0 = model.uq_model(version="v0")
uq_v2 = model.uq_model(version="v2")
```

## Classification is different

`uq_version` is regression-only — a classifier ensemble yields class
probabilities, not a value with a spread. Classification uses **VGMU**
(Variance-Gated Margin Uncertainty): the margin between the top two classes
divided by the ensemble's disagreement on them, then isotonic-calibrated to
P(correct). A calibrated confidence of 0.85 means ~85% of similar validation
predictions were right.

## What confidence does not tell you

- **High confidence != correct.** It means the models and neighbors agree, not
  that they are right.
- **It is relative to the training set.** 0.9 on a kinase solubility model says
  nothing about a PROTAC dataset. Ranks compare *within* a model, not across.
- **Novel chemistry can get falsely high confidence** if the models extrapolate
  consistently. `knn_distance` (v1) and the v2 AD score are the best guards.
- **Conformal coverage assumes exchangeability** — it degrades out of
  distribution.
- **Calibration has training-exposure bias.** Calibration std comes from running
  all 5 members over the full training set, so every row was seen by 4 of 5.
  Workbench defaults to scaffold-based splits (Bemis-Murcko) for any dataset
  with SMILES; `hyperparameters={"split_strategy": "butina"}` is stricter.

Wide intervals or low confidence usually mean **out of applicability domain**,
not a broken model. When a user asks about a specific compound, report the
interval alongside the point prediction rather than averaging uncertainty away.

## More

- https://supercowpowers.github.io/workbench/blogs/model_confidence/
