# Remove the Shared Training View + Sample-Weight State

## Motivation

Workbench batch jobs intermittently fail (~1 in 50 overnight) with:

```
INVALID_VIEW: Failed analyzing stored view '...___training':
Table '..._...___sample_weights' does not exist
```

### Root cause

Sample weights and the train/holdout split live as **shared mutable state on the
FeatureSet**:

- `_<base>___sample_weights` — a supplemental weights table
- `<base>___training` — a view that JOINs the base table against that weights table

`set_sample_weights()` mutates this state by **delete-then-recreate** with a
`time.sleep(3)` gap (`_delete_weights_table`). Promoted models (xgb, pytorch,
chemprop) all train off the *same* FeatureSet and each call `set_sample_weights()`
at startup. When two jobs run concurrently, one job's delete/recreate window
overlaps another job's read of the shared `___training` view → `INVALID_VIEW`.

The crash is the visible symptom. The deeper problem: **sample weights and the
holdout split are per-model-training concerns modeled as shared FeatureSet state.**
A silent failure mode also exists — Job B overwriting Job A's weights between A's
write and A's read means A can train on the wrong weights with no error.

### Compounding: the split has aged out

The `training` view was an 80/20 split from when "the split" was a single global
property of a dataset. Modern models do their own internal scaffold/butina/random
split via `get_split_indices()`. The FeatureSet's `training` flag and the model's
actual internal split have **already diverged**, so any "holdout" metrics derived
from the shared view are already partially leaked.

## Design Principles

1. **FeatureSet = pure data abstraction.** Columns, rows, types, id, event_time.
   No training/holdout/weight state.
2. **Splits and weights are per-model-training concerns.** They belong to a
   specific training run, owned by `FeaturesToModel`, materialized as per-model
   artifacts (`_<base>___<model>_*`).
3. **Holdouts are computed on demand, not persisted.** Either user-managed or
   temporal (`ts_inference` already recomputes the temporal split statelessly).
4. **Minimize API surface.** Backward compatibility is secondary to simplification.

## Background: AWS Feature Store types

Feature Store supports only three scalar types — `Integral`, `Fractional`,
`String` (plus `List`/`Set`/`Vector` collections). No native bool or datetime.
Irrelevant to this change directly, but context for why the supplemental-table
pattern exists.

---

## What Changes

### Shared vs per-model (the core distinction)

| Artifact | Status |
| --- | --- |
| `<base>___training` (shared view) | **REMOVE** |
| `_<base>___sample_weights` (shared table) | **REMOVE** |
| FeatureSet weight-mutation methods | **REMOVE** |
| Per-model training view (`workbench_training_view`) | **KEEP** — rebuilt from `transform(sample_weights=...)` |
| Per-model weights table | **KEEP** — now **sparse** |

### Sample weights: new flow

```python
sample_weights = fs.temporal_split(date_col, end_date)   # pure, returns {id: weight}
# compose freely from multiple producers:
#   {**outlier_weights, **temporal_weights, ...}
transform(sample_weights=sample_weights)
```

- `sample_weights` is a **sparse** dict (or `[id, weight]` DataFrame): default
  weight is `1.0`, only exceptions are stored.
- `FeaturesToModel.transform(sample_weights=...)` writes the per-model weights
  table and builds the per-model view directly — no shared state, no race.
- Per-model weights table columns: `[id, sample_weight]` (the old `training`
  column is dropped; the model does its own internal split).
- Per-model view: `base LEFT JOIN weights ... COALESCE(w.sample_weight, 1.0)`.

### Holdout / evaluation

- **User-managed:** user splits their data, trains on train, calls
  `end.inference(test_df)`.
- **Temporal:** `fs.temporal_split()` produces weights for the cutoff;
  `end.ts_inference()` recomputes the temporal holdout statelessly at eval time
  (already implemented — no change).

### Metrics: unified `default_inference_run()`

The "auto" capture priority is currently implemented three inconsistent ways
(`model_details` plugin, `confusion_matrix`, `performance_metrics`). Consolidate
into one resolver on `ModelCore`:

```
default_inference_run() -> str | None
    priority: full_cross_fold -> test_inference -> first inference run -> None
```

- `confusion_matrix`, `performance_metrics`, and the web plugins all call it.
- `model_training` (regexp-scraped from AWS logs, effectively unused) is **not**
  in the chain.
- Rename the `"auto"` capture sentinel to `"default"` to match the resolver name.

### `auto_inference` -> `test_inference`

- Method `endpoint.auto_inference()` -> `endpoint.test_inference()`; body changes
  to pull N random rows from the FeatureSet (smoke test that the endpoint serves),
  captured under `"test_inference"`.
- All capture-name lookups `"auto_inference"` -> `"test_inference"`.

### Model templates

Remove the `if n_folds == 1 and "training" in df.columns:` branch in all six
templates (chemprop, xgb, scikit_learn, ngboost, bayesian_ridge,
gaussian_process). They fall through to `get_split_indices()`, which already
handles `n_splits=1` for random / scaffold / butina (verified).

---

## Decisions Log

| # | Decision | Resolution |
| --- | --- | --- |
| D1 | `temporal_split` location/shape | Stays on FeatureSet, **pure**, returns `{id: weight}` dict (was `list` of holdout ids); no persistence |
| D2 | Sample weights ownership | Per-model input to `transform(sample_weights=...)`; FeatureSet weight-mutation methods removed |
| D3 | `auto_inference` | Rename method -> `test_inference()` (N random rows); rename capture string everywhere |
| D4 | `full_inference` | **Unchanged** — uses `model.training_view()` (per-model view), which legitimately reflects outlier removal |
| D5 | `model_training` metrics scrape | **Leave for now**; just keep it out of `default_inference_run()` |
| M1 | `model.training_view()` fallback | If per-model view missing: log **error** + create a default per-model view (all rows, weight 1.0); drop the shared-view fallback |
| M2 | Weight data structure | **Sparse** — default 1.0, store only exceptions; sparse per-model table; `LEFT JOIN ... COALESCE(., 1.0)` |

---

## Implementation Plan (ordered to keep the tree runnable)

1. **`FeaturesToModel.transform(sample_weights=...)`** — add the parameter; write
   the sparse per-model weights table from it; build the per-model view via
   `LEFT JOIN ... COALESCE`. Stop snapshotting `fs.view("training")`. Drop the
   `training` column references (features_to_model.py 138, 159, 346, 372).

2. **`fs.temporal_split()`** — make pure: return `{id: weight}` dict, remove the
   `set_sample_weights`/`add_filter` persistence.

3. **`model.training_view()`** — M1 fallback (error + create default per-model
   view); remove the line-710 shared-view fallback.

4. **Metrics consolidation** — add `ModelCore.default_inference_run()`; route
   `confusion_matrix`/`performance_metrics` "auto" -> "default" through it;
   repoint web plugins (`model_details`, `model_plot`, `confusion_explorer`).

5. **`auto_inference` -> `test_inference`** — rename method + capture string in
   `endpoint_core`, `async_endpoint_core`, `model_core` defaults,
   `shapley_values`, and the web plugins.

6. **Model templates** — remove the `"training"`-column split branch in all six.

7. **`pandas_to_features`** — remove the `incoming_hold_out_ids` flow
   (319-325, 431-432) and the `set_training_holdouts` call.

8. **Delete `TrainingView`** — `views/training_view.py`, the export in
   `views/__init__.py`, the `view_name == "training"` auto-create case in
   `view.py` (288-293), and the shell registration in `workbench_shell.py:287`.

9. **Delete FeatureSet methods** — `set_sample_weights`, `get_sample_weights`,
   `add_filter`, `set_training_holdouts`, `get_training_holdouts`,
   `_create_weights_table`, `_delete_weights_table`, the `pull_training_data`
   helper (line 374), and the `view("training")` usages.

10. **Stragglers** — `endpoint_utils.get_training_data` / `get_evaluation_data`
    (remove; they read `fs.view("training")`); `api/endpoint.py:270`.

After step 6 no library code reads the shared training view; after step 7 no
ingest path writes it; steps 8-10 are pure deletion.

## Out of Scope (deferred — downstream consumers)

- Updating downstream model pipelines (~40 call sites using `fs.set_sample_weights`).
- Downstream eval taxonomy / capture-name conventions for nightly vs promoted holdouts.
- Whether nightly eval should move to temporal (`ts_inference`) as canonical.
- D5 full removal of the `model_training` metrics scrape.

## Risks / Notes

- Existing FeatureSets in production have a shared `___sample_weights` table and
  `___training` view. New code ignores them; they become orphaned until the
  FeatureSet is recreated or cleaned up. `list_supplemental_data_tables` enumerates
  by `_<base>___*` prefix, so FeatureSet deletion still sweeps them.
- Per-model weights tables/views are swept by the same prefix mechanism on
  FeatureSet delete and by model deletion — verify both paths.
