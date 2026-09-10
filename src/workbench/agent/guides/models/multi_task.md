# Multi-Task

> building multi-task models, and deciding which auxiliary targets are worth adding

One model, several targets, one shared encoder. The encoder learns from every target;
each head keeps its own scale. Use it when the per-target data is small and the targets
are related — the auxiliaries shape the representation even where the primary is missing.

Targets are NaN where not measured. That is the multi-task shape: one row per compound,
a column per target, and the framework masks the loss for whatever is absent.

## Building the data

```python
from workbench.utils.multi_task import combine_multi_task_data, pull_multi_task_data, validate_multi_task_data

df = combine_multi_task_data([primary_df, aux_df], target_columns=[["logs"], ["logd"]], id_column="id")
validate_multi_task_data(df, target_columns=["logs", "logd"], id_column="id")
```

- **`combine_multi_task_data`** — single-task frames to one wide frame, primary first.
  Collapses rows by merge key, taking the first non-NaN per column, so a molecule in
  several sources gets all its targets on one row. `merge_on_smiles=True` joins external
  data with no shared id namespace.
- **`pull_multi_task_data`** — the same, sourced from FeatureSets rather than frames.
- **`validate_multi_task_data`** — null/duplicate ids, missing smiles, empty targets,
  featureless rows. Run it before building; the failures it catches are silent otherwise.

Convention: **the first target is the primary.** `task_weights` and `MultiTaskAlignment`
both assume it, and a multi-target endpoint reports its bare `prediction` column as
target[0].

## Building the model

`target_column` takes a list:

```python
from workbench.utils.multi_task import compute_inverse_count_task_weights

weights = compute_inverse_count_task_weights(df, target_columns=["logs", "logd"])
model = fs.to_model(
    name="multi-task-reg",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=["logs", "logd"],
    hyperparameters={"task_weights": weights, "uq_version": "v1"},
)
```

`compute_inverse_count_task_weights` weights each task by the inverse of its label count,
so a sparse target is not drowned out by a dense one. Auxiliaries usually take a fraction
of the mean primary weight rather than a weight of their own.

## Deciding which auxiliaries to add

`MultiTaskAlignment` scores each candidate before you spend a training run:

```python
from workbench.algorithms.dataframe.multi_task_alignment import MultiTaskAlignment

mta = MultiTaskAlignment(df, primary="logs", auxiliaries=["logd", "logp"], id_column="id")
mta.summary()      # one row per aux: counts, correlation, coverage, verdict
mta.results()      # per compound: UMAP coords, tanimoto_to_primary, residual_<aux>
mta.neighbors("compound_id")
```

Two mechanisms decide whether an aux helps, and `summary()` scores both:

| region | column | what it means |
|---|---|---|
| overlap | `spearman_r`, `overlap` | where both are measured, do the targets agree |
| extension | `n_aux_only`, `extension` | does the aux add chemistry the primary lacks |
| both | `recommendation` | `Use` / `Marginal` / `Risky` / `Skip` |

Reading the numbers:

- **Verdicts run on `spearman_r`, and on its magnitude.** A shared encoder exploits any
  monotone relationship, and the head's own weights carry the sign — so an efficacy or
  fold-change readout that falls as potency rises scores like a potency readout. `pearson_r`
  is reported alongside; a large gap between them means the relationship is monotone but
  nonlinear, which is fine.
- **`spearman_r` near zero is the real warning.** Two targets that don't track each other
  on shared compounds give the encoder conflicting gradients. Check `r_confidence` first —
  a correlation over 20 shared compounds is not a measurement.
- **`tanimoto_coverage_mean`** is the mean similarity from aux-having rows to their nearest
  primary-having row. Rows that have the primary count as 1.0, so an aux that mostly
  overlaps the primary scores near 1.0 by construction — compare it across extending
  auxiliaries, not against overlapping ones.
- **`residual_abs_mean`** asks whether the local SAR neighborhood predicts the aux, in std
  units. High means the aux disagrees with the chemistry around it.

An aux can be worth adding through either mechanism: well-correlated on the overlap, or
poorly overlapping but bringing chemistry the primary never saw. `Risky` usually means the
overlap disagrees while the extension looks attractive.

## What it does not tell you

Alignment scores the *data*. It does not predict that a head will help, and a well-aligned
aux can still leave the primary unchanged — auxiliary heads shape the encoder, and each
head keeps its own scale, so information about the primary's output range does not reach
it through an aux. If the primary's predictions need to cover a range its own labels never
show, that range has to be in the primary column.

Confirm any expected gain against `Model.get_inference_predictions()` on the cross-fold
capture, not against the alignment score.
