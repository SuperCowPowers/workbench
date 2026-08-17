# Local Models

!!! tip inline end "No AWS Required"
    Local artifacts need no AWS account, no config, and no credentials. Paired with `PublicData`, you can go from install to a trained model without touching AWS at all.

The Local classes mirror the Workbench artifact API against your filesystem. The
chain is the same — `DataSource → FeatureSet → Model → Endpoint` — and training
runs the same generated model script that SageMaker runs, as a subprocess. So a
script written locally publishes to AWS and produces the same model.

Storage lives under `WORKBENCH_LOCAL_PATH` (default `~/.workbench/local`).

## When to use it

Local is where you iterate: try a feature list, a framework, a set of
hyperparameters. There's no build cost and deleting is instant. AWS is where a
model lands once it's worth keeping — that's where the metrics, monitoring,
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
    target_column="Solubility",
    feature_list=["MolWt", "MolLogP", "TPSA", "NumRotatableBonds"],
)

print(model.oof_predictions())
predictions = model.to_endpoint().inference(df.head(10))
```

A LocalEndpoint isn't deployed anywhere. `inference()` loads the model in-process
through the same `model_fn`/`predict_fn` a real endpoint container uses, so the
predictions match what a deployed endpoint returns.

`validation_ids`, `sample_weights`, and `exclude_ids` work as they do in AWS, and
they're recorded so publishing can replay them.

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

Local covers training and scoring. Model metrics and plots, the inference store,
monitoring, contests, and promotion are all properties of published artifacts —
when you want those, publish.

::: workbench.local.local_data_source

::: workbench.local.local_feature_set

::: workbench.local.local_model

::: workbench.local.local_endpoint

::: workbench.local.local_meta
