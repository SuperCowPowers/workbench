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
`exclude_ids` — those are recorded and replayed on publish. A LocalEndpoint isn't
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

Training runs as a subprocess and **defaults to `wait=True`**, which blocks until it
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

Feature endpoints are the other long step, and the 3D/xTB ones are async. Check
`Endpoint(name).is_async()` rather than inferring from the name — an async endpoint
queues work instead of answering inline, so a pass over thousands of molecules runs
long enough that it belongs as its own deliberate step the user starts, not something
buried mid-chain. Say how many molecules are going through before starting one.

## Molecular features

Chemprop takes `feature_list=["smiles"]` and needs no featurization pass. Descriptor
models (XGBoost, PyTorch) do, and **the feature endpoint is the better source when one
is deployed** — it is exactly what a published model will train on, which is the whole
point of building locally first:

```python
from workbench.api import Endpoint
from workbench.api.inference_cache import InferenceCache

cached = InferenceCache(Endpoint("smiles-to-2d-v1"), cache_key_column="smiles")
local_ds = LocalDataSource(cached.inference(df), name="cyp_2d_local")
```

Featurizing locally works too, but the endpoints' `predict_fn` is **not** just the
descriptor call — it standardizes first, and skipping that yields different features
with no error and no warning:

```python
from workbench.utils.chem_utils.mol_descriptors import compute_descriptors
from workbench.utils.chem_utils.mol_standardize import standardize

df = compute_descriptors(standardize(df, extract_salts=True))  # mirror predict_fn
```

The complexity guards that skip pathological molecules live in the descriptor code, so
they apply either way — that is not a reason to prefer one path.

3D is where the wrapping matters. The xTB leg is expensive, and `InferenceCache` is an
opt-in client-side wrapper rather than something the endpoint carries, so **wrap a 3D
endpoint with it** or every run recomputes conformers it has already seen:

```python
cached = InferenceCache(Endpoint("smiles-to-3d-v2"), cache_key_column="smiles", output_key_column="orig_smiles")
```

Key the cache on a column the endpoint does not rewrite. Standardization canonicalizes
tautomers, so `smiles` comes back changed and the original is preserved as
`orig_smiles` — keying on the output column is what keeps hits across runs.

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
- No plots, inference store, monitoring, contests, or promotion. When the user
  wants those, they want a published model.
