# Endpoints and Inference

> endpoint creation, inference, captures, async, DataFrame contract

An Endpoint serves a Model. Inference runs on the Endpoint, but the results are
stored on the **Model** — see the Model/Endpoint interaction note in
`making_models`.

## DataFrame in, DataFrame out

Every Workbench endpoint follows the same contract: pass a DataFrame, get a
DataFrame back. A predictor endpoint returns your rows with prediction (and
confidence/quantile) columns appended — the output is a superset of the input,
so the columns you sent in are still there to join on.

Because the contract is uniform, endpoints chain: a feature endpoint's output is
exactly what a predictor endpoint takes as input. See the `feature_endpoints`
guide.

## Creating

```python
end = model.to_endpoint(name="my-model", tags=["my-model"])   # serverless, scales to zero
```

- **Serverless** (default) — the right choice almost always. No cost when idle.
- **Realtime** (`serverless=False`) — persistent compute, bills continuously.
  Confirm with the user before creating one.
- **Async** (`async_endpoint=True`) — conceptually a *batch* endpoint. S3 for
  I/O, up to 60-minute per-invocation timeouts, scales 0 → max_instances → 0.
  Use it for slow per-row work that would blow a normal request timeout.

## Scoring a new model

After creating an endpoint, run both:

```python
end.test_inference()          # smoke test on a sample of rows
end.cross_fold_inference()    # full cross-fold metrics
```

`test_inference()` proves the endpoint answers at all; it is not an evaluation.
`cross_fold_inference()` is what gives you trustworthy metrics. It **pulls the
out-of-fold predictions computed at training time** from S3 — it does not re-run
the folds through the endpoint — so it's quick and its cost doesn't scale with
the set size. Both are inline calls; neither belongs on Batch.

## Inference on your own data

```python
df = end.inference(eval_df)                             # not persisted
df = end.inference(eval_df, capture_name="my_holdout")  # saved to the Model
```

`capture_name` is the switch that persists a run. With it, predictions and
metrics are written and the run appears in `model.list_inference_runs()`; without
it you just get the DataFrame back. Name captures for what they are — the name
is how the user finds the run later.

The saved capture is column-trimmed, so if the user needs extra columns
alongside predictions, keep the returned DataFrame rather than re-reading the
capture.

### Where the evaluation data comes from

`test_inference()` doesn't hold a copy of any data — it **backtracks the artifact
chain** to find it:

```
Endpoint -> get_input() -> Model -> get_input() -> FeatureSet -> pull_dataframe()
```

Use the utilities rather than re-deriving that walk:

```python
from workbench.utils.endpoint_utils import backtrack_to_fs, get_evaluation_data

fs = backtrack_to_fs(end)        # the FeatureSet behind this endpoint
df = get_evaluation_data(end)    # that FeatureSet, pulled as a DataFrame
```

This is the starting point for a custom capture: backtrack to the FeatureSet,
filter the rows you want (a split column, a date range, a compound list), then
run `inference(subset, capture_name="...")`.

```python
df = get_evaluation_data(end)
subset = df[df["split"] == "phase1_test"]
results = end.inference(subset, capture_name="phase1_test")
```

When the rows you want are defined by how the model was *trained* (a `split`
column, sample weights, the validation flag), go through the model's own
training view instead — it has those columns:

```python
model = Model(end.get_input())
df = model.training_view().pull_dataframe()
subset = df[df["split"] == "phase1_test"]
```

Never hunt for a view name on the FeatureSet; `model.training_view()` resolves
it. See the views note in `data_and_features`.

Other flavors:

```python
end.full_inference()                      # everything in the model's training view
end.ts_inference(date_column, after_date) # temporal holdout (rows after a date)
end.fast_inference(eval_df, threads=4)    # no checks, no capture -- just speed
```

`fast_inference()` skips sanity checks and error handling. Use it for bulk
scoring where you already trust the input; never for a first run against new
data.

## Inspecting

```python
end.details()
end.input_columns()          # what the endpoint expects
end.output_columns()         # what it returns
end.instance_counts()        # {} for serverless -- no instances to report
```

If inference fails on column mismatch, compare `end.input_columns()` against
your DataFrame before guessing at the cause.

## Async housekeeping

```python
end.purge_async_queue()   # drop staged inputs from an abandoned backlog
```

Only valid on async endpoints; raises on sync ones. Useful when a long-running
client was killed and the fleet is still draining orphaned work.

## More

- Endpoint API: https://supercowpowers.github.io/workbench/api_classes/endpoint/
