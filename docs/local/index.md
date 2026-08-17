# Local Models

!!! tip inline end "No AWS Required"
    Local artifacts need no AWS account, no config, and no credentials. Paired with `PublicData`, you can go from install to a trained model without touching AWS at all.

The Local classes mirror the Workbench artifact API against your filesystem. The
chain is the same — `DataSource → FeatureSet → Model → Endpoint` — and training
runs the same generated model script that SageMaker runs, as a subprocess. So a
script written locally publishes to AWS and produces the same model.

Storage lives under `WORKBENCH_LOCAL_PATH` (default `~/.workbench/local`).

Local is where you iterate: try a feature list, a framework, a set of
hyperparameters. There's no build cost and deleting is instant. AWS is where a
model lands once it's worth keeping — that's where monitoring, deployed
endpoints, and everything else the team consumes live.

## Try it

`PublicData` reads public S3 anonymously, so this runs with no AWS setup:

```python
from workbench.local import LocalDataSource, PublicData, ModelType, ModelFramework

df = PublicData().get("comp_chem/aqsol/aqsol_public_data")

ds = LocalDataSource(df, name="aqsol_local")
fs = ds.to_features("aqsol_local_features", id_column="ID")
model = fs.to_model(
    "aqsol-local-reg",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.XGBOOST,
    target_column="solubility",
    feature_list=["molwt", "mollogp", "tpsa", "numrotatablebonds"],
)

print(model.get_inference_metrics())
predictions = model.to_endpoint().inference(fs.pull_dataframe().head(10))
```

`to_features()` lowercases column names, same as the AWS path, so `target_column`
and `feature_list` refer to the FeatureSet's names rather than the source frame's.
Use `fs.columns` to see them.

A LocalEndpoint isn't deployed anywhere. `inference()` loads the model in-process
through the same `model_fn`/`predict_fn` a real endpoint container uses, so the
predictions match what a deployed endpoint returns.

`validation_ids`, `sample_weights`, and `exclude_ids` work as they do in AWS, and
they're recorded so publishing can replay them.

The Workbench REPL exposes all of these, so none of the imports are needed there.

## Scoring

Inference runs work the same as they do on an AWS Model, so a script that walks
them runs against either. Metrics are computed from the run's predictions.

```python
model.list_inference_runs()                     # ["full_cross_fold", ...]
model.get_inference_metrics()                   # defaults to full_cross_fold
model.get_inference_predictions("full_cross_fold")

model.oof_predictions()          # the cross-fold predictions directly
model.validation_predictions()   # held-out rows, when validation_ids were used

# Naming a capture adds it to the run list
model.to_endpoint().inference(eval_df, capture_name="holdout")
model.get_inference_metrics("holdout")
```

The `model_training` run an AWS Model carries has no local equivalent — those
metrics come from SageMaker scraping the training job's output.

## Publishing

```python
model.publish_plan()        # what it would create, creates nothing
aws_model = model.publish() # ds -> fs -> model -> endpoint
```

Publishing walks up the lineage and creates whatever AWS doesn't already have,
then deploys an endpoint (pass `endpoint=False` to stop at the model). It
**retrains in AWS** from the published FeatureSet rather than uploading local
artifacts, so the model lands in the registry like any other — with the row roles
replayed, so it trains on the same rows.

That launches a real SageMaker training job, which is why `publish_plan()` is a
separate call: look before you leap.

If a published model disagrees with the local one, `model.version_drift()` reports
package versions that differ between this machine and the training image.

## Listing and deleting

```python
from workbench.local import LocalMeta

LocalMeta().models()     # also data_sources(), feature_sets(), endpoints()
```

The Workbench REPL prints a local summary at startup and on `local_summary()`.

Always delete through the API. `LocalModel.delete()` takes its endpoints with it;
removing directories by hand leaves endpoints pointing at a model that no longer
exists. That's the only cascade — deleting a LocalFeatureSet leaves its models
alone.

## What's not here

Local covers training and scoring. Plots, the inference store, monitoring,
contests, and promotion are all properties of published artifacts — when you want
those, publish.

::: workbench.local.local_data_source

::: workbench.local.local_feature_set

::: workbench.local.local_model

::: workbench.local.local_endpoint

::: workbench.local.local_meta
