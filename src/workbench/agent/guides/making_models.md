# Making Models

> build a model, deploy an endpoint, and score it

The pipeline is `DataSource -> FeatureSet -> Model -> Endpoint`. Each stage is
an artifact in AWS; each `to_*()` call launches real infrastructure.

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
model.details()             # metadata, metrics, hyperparameters
model.performance_metrics() # scored metrics (populated by the inference runs above)
model.list_inference_runs() # capture names available on this model
```

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
