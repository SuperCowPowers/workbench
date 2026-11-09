# Target Transforms

A skewed target often trains better in transformed space — `log10` on a concentration, a
solubility, a percent unbound. The transform belongs on the **model**, not on the FeatureSet:

- No FeatureSet to rebuild, and one FeatureSet feeds models with different transforms
- Metrics stay on the original scale, so a transformed model compares directly against an
  untransformed one
- A transformed model goes through promotion like any other

Training applies the transform to the target column; inference inverts it. Predictions come
back in the units you measured.

Available for XGBoost, PyTorch, ChemProp, TabICL and Atom-JEPA, regression only.

## Setting one

Two hyperparameters on an otherwise normal `to_model()` call:

```python
from workbench.api import FeatureSet, ModelType, ModelFramework

fs = FeatureSet("mppb_raw")

model = fs.to_model(
    name="mppb-xgb-log10",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.XGBOOST,
    target_column="mppb",
    feature_list=features,
    hyperparameters={
        "target_transform": {"name": "log10", "offset": 0.1},
        "prediction_bounds": [0.0, 100.0],
    },
)

end = model.to_endpoint()
end.inference(eval_df)   # percent unbound, not log space
```

| hyperparameter | default | what it does |
|---|---|---|
| `target_transform` | `None` (identity) | the map applied to the target during training and inverted at inference |
| `prediction_bounds` | `None` | `[low, high]` clip on the predictions, applied after the inverse |

The model stores what it was fitted with, and inference reads that back — so the inverse is
always exactly the forward that trained it, never a separately-passed value that can drift.

## log10 and its offset

`log10` is `log10(y + offset)`, and `offset` defaults to 0. A bare `"log10"` is therefore a
plain `log10(y)`; `log10(y + 1)` is written out:

```python
"target_transform": "log10"                             # log10(y)
"target_transform": {"name": "log10", "offset": 1.0}    # log10(y + 1)
```

**The offset belongs just above the assay's reporting floor** — the smallest value the assay
actually reports. Below that floor the assay cannot separate values, so resolution spent
there is spent on noise.

Zeros pin it down more precisely. A target containing 0 raises without an offset, and those
zeros are normally below-quantitation rows; the offset decides where they land relative to
the real readings. Keep them within **a few reporting steps** of the floor — near enough to
read as "under the lowest measurement," rather than the ten or more steps that turn a handful
of rows into a detached cluster for the loss to chase.

Both numbers come out of the data:

```python
import numpy as np

y = fs.pull_dataframe()["mppb"]
readings = np.sort(y[y > 0].unique())
floor = readings[0]                                   # smallest real reading
step = np.median(np.diff(np.log10(readings[:10])))    # typical step, in log units
```

The same rule lands on very different offsets, because assays report to different precision:

| reporting floor | typical step | offset | zeros sit |
|---|---|---|---|
| 0.1 (MPPB) | 0.079 log units | 0.1 | 3.8 steps below the floor |
| 0.002 | 0.125 log units | 0.005 | 1.2 steps below the floor |

A 50x difference in offset for two readings of the same quantity, so copying a number from
another model is the one reliable way to get this wrong.

As a sanity check, an offset large relative to the target's **range** stops being a transform
at all. Share of the transformed output range given to the bottom decile of the data, where a
linear map gives 10%:

| target range | offset 1.0 | offset 0.1 | offset 0.01 |
|---|---|---|---|
| 0 – 1 | 13.8% | 28.9% | 52.0% |
| 0 – 87.6 (MPPB) | 50.8% | 66.2% | 74.6% |

So `offset=1.0` over a fraction is barely a transform at all, while the same offset over a
percentage is a strong one.

## Prediction bounds

`prediction_bounds` is independent of any transform — a non-negative or percentage target
wants it under the identity transform too. Either side may be `None` to leave it unbounded.

Clipped values pile up as a mass point at the bound, which destroys ordering among them, so
bound only targets whose predictions should not reach the bound.

## Multi-target models

A single spec applies to every target. Since a multi-target model usually carries targets in
different units, a `{target_column: spec}` mapping sets them one at a time, and targets it
omits get identity:

```python
hyperparameters={
    "target_transform": {"ksol": "log10", "mppb": {"name": "log10", "offset": 0.1}},
    "prediction_bounds": {"mppb": [0.0, 100.0]},
}
```

## What inference gives back

| | |
|---|---|
| point prediction | `inverse(mean)` — the conditional median, which is what MAE scores |
| `prediction_std` | `std` of the inverted ensemble members, so the spread is in original units |
| metrics | always the original scale |
| rank metrics | identical with and without a transform |

That last row is the contract: the transform is monotonic, so Spearman and Kendall cannot
move. Only placement-dependent metrics — MAE, RMSE, R² — change, and they change because the
model learned something different, not because the scale shifted.

## What a transform buys

MPPB percent unbound, 1302 compounds, out-of-fold, all metrics on the original scale:

| model | MAE | RMSE | spearman | MAE under 1% |
|---|---|---|---|---|
| no transform | 6.784 | 10.630 | 0.785 | 3.210 |
| log10, offset 0.1 | 6.602 | 11.145 | 0.791 | **1.826** |

Error on the clinically interesting tail drops 43% while RMSE rises 5%. Squared error in
original space is dominated by the few high-unbound compounds; the transform spends its
resolution on the low end instead. That trade is the whole reason to reach for one — and the
reason to check the tail rather than the headline number.

## Limits

- Regression only — a transform on a classifier raises
- Not available with a scikit-learn `model_class`
- Cannot be HPO-searched. The target is transformed once, before the search runs, so a
  searched transform would differ between the trials and the winner's saved config that
  inference reads back. Set it as a top-level hyperparameter and search around it.

---

## Questions?

<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)
