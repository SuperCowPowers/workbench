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
from workbench.local import DataSource, PublicData, ModelType, ModelFramework

df = PublicData().get("comp_chem/aqsol/aqsol_public_data")
local_ds = DataSource(df, name="aqsol_local")
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
`exclude_ids` — those are recorded and replayed on publish. A local Endpoint isn't
deployed anywhere; `inference()` loads the model in-process through the
container's own `model_fn`/`predict_fn`.

Score it the same way you would an AWS model. `list_inference_runs()` returns the
training cross-fold plus any endpoint captures, and metrics are computed from the
run's predictions:

```python
local_model.list_inference_runs()                    # ["full_cross_fold", ...]
local_model.get_inference_metrics()                  # defaults to full_cross_fold
local_model.get_inference_predictions("full_cross_fold")
```

## Don't block the turn

Pick the runner by how long the work runs, not by how big it looks:

| Runner | Runtime | Outlives |
|---|---|---|
| in process | seconds | nothing — it *is* the turn |
| subprocess, `wait=False` | minutes | the turn, not the REPL |
| Batch (`batch`) | hours | the session |

Local DataSources, FeatureSets, and `Endpoint.inference()` are all in process.

Training is a subprocess and **defaults to `wait=True`**, which blocks until it
finishes. Pass `wait=False` for **any** model creation, not just the ones that look
slow — polling a train that finished in seconds costs one call, while blocking on one
that runs for twenty minutes leaves the user staring at a frozen session with no way
to redirect:

```python
local_model = local_fs.to_model(..., wait=False)
local_model.training_state()   # includes "interrupted" for a run whose process died
```

`training_state()` reports `interrupted` rather than leaving a dead run claiming to be
training, so a train that never finished never reads as done.

## Molecular features

Chemprop takes `feature_list=["smiles"]` and needs no featurization pass. Descriptor
models (XGBoost, PyTorch) do, and **the rung depends on whether the features need
xTB**, not on how many molecules there are.

Whichever endpoint produced the features, `end.output_columns()` is the feature
list — it returns exactly the columns that endpoint emits, so labels, ids and
bookkeeping columns are excluded by construction.

**2D and fingerprints: the endpoint, as a subprocess job.** Around 5,000 molecules
takes ~10 minutes — too long to block a turn, too short to be worth Batch:

```python
from workbench.utils.feature_endpoint_job import run_feature_endpoint

job = run_feature_endpoint(df, "smiles-to-2d-v1", name="cyp_2d", wait=False)
job.state()      # running -> completed, same vocabulary as training
job.results()    # the featurized rows once it's completed
```

Calling the endpoint directly blocks for those same ten minutes whether or not it is
async — async is how SageMaker runs the invocation, not something the caller gets back
from.

**3D: Batch.** Conformer generation plus xTB runs at 1-2 molecules/second, so those
same 5,000 molecules take 3-4 hours, well past the life of a session. Put the pass in
a Batch script (`batch`). Its product is a warmed cache, so the features then read
back locally in seconds:

```python
from workbench.api import Endpoint

cached = InferenceCache(Endpoint("smiles-to-3d-v2")).inference(df)
```

`InferenceCache` defaults are already right for every feature endpoint — take them.
Standardization canonicalizes tautomers, so `smiles` comes back rewritten and the cache
keys on `orig_smiles` instead. Say how many molecules are going through before starting
a pass.

**With no AWS at all**, compute 2D in process — the endpoint's `predict_fn` is exactly
these two calls:

```python
from workbench.utils.chem_utils.mol_descriptors import compute_descriptors
from workbench.utils.chem_utils.mol_standardize import standardize

local_ds = DataSource(compute_descriptors(standardize(df, extract_salts=True)), name="cyp_2d_local")
```

Standardization is not optional — skipping it yields different features with no error
and no warning. `extract_salts=True` matches `smiles-to-2d-v1`, the default for every
assay; only solubility work keeps salts (`False`, matching `smiles-to-2d-salt-v1`).
There is no local 3D path; `tblite` is installed only in the images.

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

- Delete through the API. A local `Model.delete()` takes its endpoints with it;
  removing directories by hand leaves them pointing at a model that's gone.
- No plots, inference store, monitoring, contests, or promotion. When the user
  wants those, they want a published model.
