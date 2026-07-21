# Making Models

> build a model, deploy an endpoint, and score it

The pipeline is `DataSource -> FeatureSet -> Model -> Endpoint`. Each stage is
an artifact in AWS; each `to_*()` call launches real infrastructure.

## Where to start

When a user says "let's make a model", the first thing to settle is what it is
built from. **Prefer the shortest chain.** In priority order:

| Input | Chain to a model | |
|---|---|---|
| **FeatureSet** | `fs.to_model()` | **recommended** — one call |
| **DataSource** | `ds.to_features()` → `fs.to_model()` | one extra step |
| **PublicData** | `pub_data.get()` → `DataSource` → `to_features()` → `to_model()` | |
| **S3 / local file** | `DataSource(path)` → `to_features()` → `to_model()` | |

**Always check for an existing FeatureSet first:**

```python
feature_sets()        # or CachedMeta().feature_sets()
```

FeatureSet → Model is the shortest chain and the one to recommend. Reusing a
FeatureSet also keeps models comparable, since they train on the same prepared
data.

The longer chains are perfectly fine — just more work and more decisions to get
right (`id_column`, one-hot columns, naming). If the user hasn't said what to
start from, ask rather than assuming, and offer the existing FeatureSets as the
first option.

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
this costs nothing to leave up.

## Where it runs: inline vs. Batch

`to_model()` launches a SageMaker training job and then **blocks the REPL**,
polling until the training finishes. For a quick model that's fine; for a heavy
one it ties up the session for the whole train.

- **Quick** — XGBoost / sklearn on a modest set: call `to_model()` inline.
- **Heavy** — chemprop or pytorch (a real GNN / neural train), an HPO sweep, or a
  large FeatureSet: **don't run it inline.** Put the same `to_model()` call in a
  script and launch it on Batch, so the REPL stays free (see the `batch` guide):

  ```python
  from workbench.utils.batch_utils import launch_batch

  code = '''
  from workbench.api import FeatureSet, ModelType, ModelFramework
  fs = FeatureSet("mppb_features")
  fs.to_model(name="mppb-reg", model_type=ModelType.REGRESSOR,
              model_framework=ModelFramework.CHEMPROP, target_column="mppb",
              feature_list=["smiles"], hyperparameters={"uq_version": "v1"})
  '''
  job = launch_batch(code, name="mppb_reg_chemprop", size="medium")
  ```

  The Batch job runs `to_model()` in a detached container — same result (a new
  Model artifact you query afterward), but your session isn't blocked. It's
  billable compute, so confirm before launching.

## Conventions

- `id_column` is **required** on `to_features()`. If the data has no natural id,
  say so — don't invent a row index silently.
- Name models by role: `-reg` regression, `-class` classification. Endpoints are
  serverless by default; only add a `-rt` suffix for a realtime (`serverless=False`)
  endpoint.
- Set `hyperparameters={"uq_version": "v1"}` on new models — strongest
  uncertainty quantification and keeps comparisons consistent.
- Check `fs.columns` (a property) and `fs.column_details()` before choosing a
  `feature_list`. Don't guess column names.

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

- **`sample_weights`** — a `{id: weight}` dict (or a `[id_column,
  "sample_weight"]` DataFrame), forwarded to the model script as-is. Ids not
  listed default to `1.0`. It carries **no** role semantics: a weight of `0.0`
  is just a zero weight, **not** an exclusion. To drop a row, use `exclude_ids`.
- **`validation_ids`** — kept in the training view and marked, but routed out of
  training and scored as a genuine held-out set.
- **`exclude_ids`** — dropped from the training view entirely; no model ever
  sees them. Use for outliers and anomalies. On overlap it **takes precedence**
  over `validation_ids`.

## Dependencies

Artifacts form a chain: `ds -> fs -> model -> endpoint`. A FeatureSet never
feeds an Endpoint directly. Deleting an upstream artifact orphans everything
downstream.

## Cost

Not a concern for normal work. Training images are right-sized automatically,
and `to_endpoint()` is **serverless by default** (`serverless=True`) — it scales
to zero when idle. Don't warn about cost or ask for confirmation.

The one exception is a **realtime** endpoint (`serverless=False`): persistent
compute that bills continuously. Only there, confirm with the user first.

## Inspecting

```python
model.details()                          # metadata, hyperparameters
model.list_inference_runs()              # capture names available on this model
model.get_inference_metrics(capture)     # metrics for a capture (rmse, mae, r2, spearmanr, ...)
```

`get_inference_metrics(capture_name="default")` returns a one-row metrics
DataFrame (or `None` if that capture doesn't exist — check
`list_inference_runs()` first). There is no `performance_metrics()` method.

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
