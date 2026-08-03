# Making Models

> build a model, deploy an endpoint, and score it

The pipeline is `DataSource -> FeatureSet -> Model -> Endpoint`. Each stage is
an artifact in AWS; each `to_*()` call launches real infrastructure. The chain is
strict — a FeatureSet never feeds an Endpoint directly — and deleting an upstream
artifact orphans everything downstream.

## Where to start

When a user says "let's make a model", the first thing to settle is what it is
built from. **Prefer the shortest chain.** In priority order:

| Input | Chain to a model | |
|---|---|---|
| **FeatureSet** | `fs.to_model()` | **recommended** — one call |
| **DataSource** | `ds.to_features()` → `fs.to_model()` | one extra step |
| **DataFrame** | `PandasToFeatures()` → `fs.to_model()` | incl. `pub_data.get()` |
| **S3 / local file** | `DataSource(path)` → `to_features()` → `to_model()` | |

**Always check for an existing FeatureSet first:**

```python
feature_sets()        # or CachedMeta().feature_sets()
```

Reusing a FeatureSet also keeps models comparable, since they train on the same
prepared data. The longer chains are fine — just more decisions to get right
(`id_column`, one-hot columns, naming). If the user hasn't said what to start
from, ask rather than assuming, and offer the existing FeatureSets first.

## Flow

```python
ds = DataSource("s3://bucket/data.csv", name="my_data")
fs = ds.to_features("my_features", id_column="id")
model = fs.to_model(name="my-model-reg", model_type=ModelType.REGRESSOR,
                    target_column="solubility", feature_list=features)
end = model.to_endpoint(name="my-model", tags=["my-model"])   # serverless by default

# Always deploy an endpoint after a model, then score it:
end.test_inference()          # quick sanity check on held-out rows
end.cross_fold_inference()    # full cross-fold metrics (capture: full_cross_fold)
```

A model isn't "done" until it has an endpoint and both inference runs — that's
what populates its metrics and predictions. Since the endpoint is serverless,
this costs nothing to leave up. **This is the deliverable regardless of where
training runs** — inline or on Batch, the full chain (`to_model` → `to_endpoint`
→ `test_inference` → `cross_fold_inference`) is what makes a usable model.

## Where it runs

Every `to_*()` above is a create: ask first, then default to Batch (`batch`).
`to_model()` blocks the REPL until the SageMaker training job finishes, so a
heavy train ties up the whole session. On Batch, put the **whole chain** in the
script — the process is headless, so a `to_model()` that stops there leaves a
model with no endpoint and no metrics.

## Conventions

- `id_column` is **required** on `to_features()`. If the data has no natural id,
  say so — don't invent a row index silently.
- Name models by role: `-reg` regression, `-class` classification. Endpoints are
  serverless by default; only add a `-rt` suffix for a realtime (`serverless=False`)
  endpoint.
- Set `hyperparameters={"uq_version": "v1"}` on new models — strongest
  uncertainty quantification and keeps comparisons consistent.
- Check `fs.columns` (a property), or pull a DataFrame and read `df.dtypes`,
  before choosing a `feature_list`. Don't guess column names.

## Weights, validation, and exclusions

Three **separate** `to_model()` arguments, each doing exactly one thing. Don't
try to express one through another.

```python
model = fs.to_model(
    ...,
    sample_weights={id_1: 2.0, id_2: 0.5},   # per-id weight, nothing more
    validation_ids=[id_3, id_4],             # honest held-out scoring set
    exclude_ids=[id_5],                      # dropped entirely
)
```

- `sample_weights` carries **no** role semantics: a weight of `0.0` is a zero
  weight, **not** an exclusion. To drop a row, use `exclude_ids`.
- `exclude_ids` **takes precedence** over `validation_ids` on overlap.
- `validation_ids` scoring tells you how the *model* generalizes. For the ground
  truth on *deployed* performance, run `endpoint.inference()` on those same ids —
  it exercises the real serving path end to end.

## Cost

Not a concern for normal work — training images are right-sized automatically and
endpoints are serverless by default, scaling to zero when idle. Don't warn about
cost or ask for confirmation. The one exception is a **realtime** endpoint
(`serverless=False`): persistent compute that bills continuously, so confirm
that one with the user first.

## Inspecting

See `exploring`. There is no `performance_metrics()` method.

## Model / Endpoint interaction

Inference is run on the **Endpoint** but the results land on the **Model**. So
after `end.test_inference()` or `end.cross_fold_inference()`, those runs show up
in `model.list_inference_runs()`.

Objects cache their metadata, so a Model you already have in hand will not show
a run that was created after you fetched it. Recreate it to pick up new results:

```python
end.test_inference()
model = Model("my-model-reg")     # re-fetch, otherwise the new run is missing
model.list_inference_runs()
```
