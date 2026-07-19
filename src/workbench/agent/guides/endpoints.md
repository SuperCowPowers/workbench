# Endpoints and Inference

An Endpoint serves a Model. Inference runs on the Endpoint, but the results are
stored on the **Model** — see the Model/Endpoint interaction note in
`making_models`.

**Name your handles `model` and `end`.** Variables you create stay in the user's
session, so use the names they expect: `end = Endpoint(...)` and
`model = Model(...)`. Not `e`, not `ep`, not `my_endpoint`.

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
`cross_fold_inference()` is what gives you trustworthy metrics.

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
