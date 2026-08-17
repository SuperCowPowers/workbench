# Local Models

> train models on this machine with no AWS, then publish the one that works

Local artifacts mirror the AWS API against the filesystem (`WORKBENCH_LOCAL_PATH`,
default `~/.workbench/local`). Training runs the same generated model script
SageMaker runs, as a subprocess, so what works locally publishes and produces the
same model. No config, no credentials, no cost.

## Local or AWS?

**The default follows the input, not a mode.** Don't ask.

| The user points at | Build |
|---|---|
| a CSV, a DataFrame, `PublicData()` | **local** |
| an existing local artifact | **local** |
| an existing AWS DataSource / FeatureSet / Model | **AWS** |

Publishing is the one move between them, and only the user asks for it —
"publish", "deploy this", "put it in AWS".

There is no AWS-to-local path for artifacts. To iterate on data that already lives
in AWS, start a local chain from `FeatureSet("aqsol_features").pull_dataframe()`.

## Flow

`PublicData` reads public S3 anonymously — no AWS account — and is re-exported
from `workbench.local`, so it's the usual first step.

```python
from workbench.local import LocalDataSource, PublicData, ModelType, ModelFramework

df = PublicData().get("comp_chem/aqsol/aqsol_public_data")
local_ds = LocalDataSource(df, name="aqsol_local")
local_fs = local_ds.to_features("aqsol_local_features", id_column="ID")
local_model = local_fs.to_model(
    "aqsol-local-reg",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.XGBOOST,
    target_column="Solubility",
    feature_list=["MolWt", "MolLogP", "TPSA"],
)
preds = local_model.to_endpoint().inference(eval_df)
```

Same argument names as AWS, including `validation_ids` / `sample_weights` /
`exclude_ids` — those are recorded and replayed on publish. Score with
`oof_predictions()`. A LocalEndpoint isn't deployed anywhere; `inference()` loads
the model in-process through the container's own `model_fn`/`predict_fn`.

For a long train, `wait=False` returns immediately and `training_state()` polls it.

## Publishing

```python
local_model.publish_plan()          # what it would create, creates nothing
aws_model = local_model.publish()   # ds -> fs -> model -> endpoint
```

Publishing **retrains in AWS** from the published FeatureSet, replaying the row
roles, so the model lands in the registry like any other. It launches a real
SageMaker training job — show `publish_plan()` and confirm first. If a published
model disagrees with the local one, check `version_drift()`.

## Watch for

- Delete through the API. `LocalModel.delete()` takes its endpoints with it;
  removing directories by hand leaves them pointing at a model that's gone.
- No metrics, plots, inference store, monitoring, contests, or promotion. When the
  user wants those, they want a published model.
