# Target Transforms

> training on a transformed target: log10, logit, arcsinh, prediction_bounds

A skewed regression target often trains better in transformed space. The
transform lives on the **model**, as a hyperparameter: training applies it to
the target, inference inverts it. Predictions and metrics are always in the
original units, so a transformed model compares directly against an
untransformed one. Full write-up with figures and measured examples:
`docs/models/target_transforms.md`.

Works for XGBoost, PyTorch, ChemProp, TabICL and Atom-JEPA; regression only.

| transform | map | fits when the target |
|---|---|---|
| `log10` | `log10(y + offset)` | spans decades and stays positive |
| `logit` | `log10(y / (ceiling - y))` | is bounded at both ends (a percent) |
| `arcsinh` | `asinh(y / scale)` | has exact zeros or negatives |

## Setting one

```python
hyperparameters={
    "target_transform": "log10",                              # bare name
    "target_transform": {"name": "log10", "offset": 0.1},     # with arguments
    "target_transform": {"name": "logit", "ceiling": 120.0},
    "target_transform": {"name": "arcsinh", "scale": 0.1},
    "prediction_bounds": [0.0, 100.0],                        # either side may be None
}
```

- Defaults: `offset` 0, `scale` 1.0; `logit` needs a `ceiling`.
- `prediction_bounds` clips predictions and UQ quantiles after the inverse, and
  is independent of the transform. A `logit` model is bounded by construction.
- Multi-target: a `{target_column: spec}` mapping for either hyperparameter;
  targets left out get identity.
- `model.target_transform()` returns what a model was fitted with.

## Choosing the parameters

`offset` and `scale` are the same knob: where resolution stops being spent. The
assay's smallest reported value is the natural place for it, and it varies a lot
between assays, so it is worth reading from the data:

```python
y = fs.pull_dataframe()[target]
floor = y[y > 0].min()
```

A `logit` ceiling above the physical bound keeps every measured value in
domain; the closer it sits to the bound, the more range the top tail gets.

## Domains

`to_model()` checks the spec against the FeatureSet's cached min/max and raises
before any training job starts. The error names the fix.

| transform | rejects | fix |
|---|---|---|
| `log10` | `y + offset <= 0` | a larger offset |
| `logit` | `y <= 0` | clamp or drop those rows in the FeatureSet |
| `logit` | `y >= ceiling` | a higher ceiling, or cap the rows |
| `arcsinh` | nothing | |

## What changes and what doesn't

- Every map is monotonic, so rank metrics (Spearman, Kendall) are identical with
  and without one. MAE, RMSE and R² move because the model learned differently.
- The usual trade is better error in the low tail for somewhat worse RMSE, so
  the tail is the place to look when judging whether it helped.
- `prediction_std` is the spread of the inverted ensemble members, in original
  units, and is not clipped by `prediction_bounds`.

## Limits

- A transform on a classifier, or with a scikit-learn `model_class`, raises.
- Not HPO-searchable: it is a top-level hyperparameter, with the search run
  around it.
