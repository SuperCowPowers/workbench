# Local Mode

!!! tip inline end "No AWS Required"
    Local artifacts need no AWS account, no config, and no credentials. Paired with `PublicData`, you can go from install to a trained model without touching AWS at all.

`workbench.local` mirrors the Workbench artifact API against your filesystem. The
chain is the same — `DataSource → FeatureSet → Model → Endpoint` — and training
runs the same generated model script that SageMaker runs, as a subprocess. So a
script written locally publishes to AWS and produces the same model.

Storage lives under `WORKBENCH_LOCAL_PATH` (default `~/.workbench/local`).

Local is where you iterate: try a feature list, a framework, a set of
hyperparameters. There's no build cost and deleting is instant. AWS is where a
model lands once it's worth keeping — that's where monitoring, deployed
endpoints, and everything else the team consumes live.

## Starting from nothing

With no AWS config, the REPL starts in **local mode**:

```bash
pip install workbench
workbench
```

The prompt comes up with the local classes and `pub_data` already bound, so the
example below runs as-is with no imports. When you're ready to connect an AWS
account, run `aws_setup()` from that prompt.

## The agent

[Bosco](../bosco/index.md) runs here too. With no AWS account it needs an Anthropic
API key to reach a model:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
workbench
```

`Bosco` then appears in the prompt and you can ask for a model in English rather
than writing the chain yourself. The status line names where prompts go — `status`
shows `Bosco: your AWS account` on a connected session and `Bosco: Anthropic API
key` here — so it is always visible whether your data leaves the machine.

## Try it

`PublicData` reads public S3 anonymously, so this runs with no AWS setup:

```python
from workbench.local import DataSource, PublicData, ModelType, ModelFramework

df = PublicData().get("comp_chem/aqsol/aqsol_public_data")

ds = DataSource(df, name="aqsol_local")
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

!!! warning "In the REPL, use the bound names"
    The import above is for scripts. In a REPL connected to AWS, running it rebinds
    `DataSource` to the local class for the rest of your session — the next artifact
    you build lands on disk instead of AWS.

    The names are already bound for you: in local mode the bare `DataSource` /
    `FeatureSet` / `Model` / `Endpoint` are the local classes, and in any session
    `LocalDataSource`, `LocalFeatureSet`, `LocalModel`, and `LocalEndpoint` are.

`to_features()` lowercases column names, same as the AWS path, so `target_column`
and `feature_list` refer to the FeatureSet's names rather than the source frame's.
Use `fs.columns` to see them.

A local Endpoint isn't deployed anywhere. `inference()` loads the model in-process
through the same `model_fn`/`predict_fn` a real endpoint container uses, so the
predictions match what a deployed endpoint returns.

`validation_ids`, `sample_weights`, and `exclude_ids` work as they do in AWS, and
they're recorded so publishing can replay them.

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

## Neighbors and uncertainty

A local model carries the same proximity and UQ artifacts an AWS one does, because
training writes them either way:

```python
prox = model.prox("fingerprint")        # training-time neighbors, or fresh from the FeatureSet
prox.neighbors("compound-42", n_neighbors=6)

model.uq_model()                        # the fitted UQ model, for calibrated intervals
```

`fs.prox(...)` builds one over a FeatureSet before any model exists — the pre-model
path for hunting analogs and activity cliffs.

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
from workbench.local import Meta   # in the REPL: LocalMeta

Meta().models()     # also data_sources(), feature_sets(), endpoints()
```

The Workbench REPL prints a local summary at startup and on `local_summary()`.

Always delete through the API. A local `Model.delete()` takes its endpoints with it;
removing directories by hand leaves endpoints pointing at a model that no longer
exists. That's the only cascade — deleting a FeatureSet leaves its models
alone.

## What's not here

The inference store, monitoring, contests, and promotion are properties of
published artifacts — when you want those, publish.

**Plots work.** `get_inference_predictions()` feeds parity and residual plots and
`prox()` backs neighborhood graphs, the same as an AWS model. What's missing are
the plots fed by AWS-only methods: SHAP, confusion matrix, and HPO.

**Chemprop and PyTorch need `pip install workbench[modeling]`** — the base install
carries XGBoost, RDKit, and the descriptor stack, but not torch. Local inference
loads the model in this process, and torch and XGBoost each bring their own OpenMP
runtime, so a session that predicts with chemprop can't then predict with an
XGBoost model until the REPL restarts.

There is no local 3D featurization: conformers plus xTB run in the training images
only. 2D descriptors compute in process.

::: workbench.local.data_source

::: workbench.local.feature_set

::: workbench.local.model

::: workbench.local.endpoint

::: workbench.local.meta
