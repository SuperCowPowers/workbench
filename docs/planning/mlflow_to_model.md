# MLflow → Workbench Model Onboarding (`mlflow_to_model`)

## Principle (why this exists)

Workbench is a **publishing** framework; MLflow is the open standard for
**pre-publish exploration** (local iteration, HPO trials, run comparison). They
are complementary, not competing — the same open-interface / managed-backend
split AWS and Databricks both use. This is the "published **AND** local" path:

- **MLflow owns the messy phase** — iterate fast, locally, on tabular models;
  every run tracked and diffable. None of it is a Workbench artifact.
- **Workbench owns publishing** — the one winning model becomes a durable,
  named, shared, inspectable, deployable artifact.
- **`mlflow_to_model` is the bridge** — package a trained MLflow model into a
  first-class Workbench Model, with **no SageMaker training job**.

Note MLflow is already a hard dependency of `sagemaker-train` / `sagemaker-serve`
(the v3 SDK), so it ships whether or not we use it — no new dependency.

## What a Workbench Model is (the target contract)

A Workbench Model is a SageMaker **Model Package** (a group + a versioned
package) whose `InferenceSpecification` points at:

- **`Image`** — a Workbench inference image expecting the standard serving
  harness and the **CSV feature-in / prediction-out** contract.
- **`ModelDataUrl`** — a `model.tar.gz` bundling the trained model **plus the
  Workbench inference entry point + metadata**.

`onboard()` then sets model type, framework, `feature_list`, `target`,
`id_column`, class labels, and loads training metrics/hyperparameters.

The training job is only the thing that *produces* that tarball. **Ingestion
means producing the same tarball + registration from an already-trained model.**

## Converged design

`mlflow_to_model` is a **`copy()`-shaped ingest**. `ModelCore.copy()`
(`model_core.py`) already proves the publish tail — freeze a tarball, register a
Model Package pointing at it, carry metadata as group tags, land onboarded. We
reuse that; only the head differs.

**Shared core (extracted from `copy()`):** *register a Workbench-contract
tarball at an S3 URL as model group Y, carry metadata as tags, land onboarded.*
Both `copy()` and `mlflow_to_model` call it.

| | `copy()` head | `mlflow_to_model` head |
|---|---|---|
| Source | an existing Workbench Model | an MLflow model (run id / registry uri / local `MLmodel` dir) |
| Artifact step | freeze the source `model.tar.gz` | unpack MLmodel → extract raw model by flavor → **rewrap in Workbench harness** → upload |
| Metadata | inherited from source model | **sourced from MLmodel + caller** (below) |

### The contract decision: rewrap, don't adopt

The ingested model is **rewrapped into Workbench's inference harness** (extract
the raw booster/estimator, drop it into the existing framework harness layout,
tar). We deliberately **do not** deploy the raw MLmodel under an MLflow-flavor
container.

- **Rewrap (chosen).** First-class Workbench Model: `test_inference`,
  `cross_fold_inference`, capture-to-model metrics, UQ columns, feature-skew
  protection all work, because the endpoint speaks Workbench's contract.
- **Adopt MLflow container (rejected).** "Onboard *any* MLflow model" for near-
  zero work, but the endpoint is MLflow-shaped and doesn't speak Workbench's
  inference/capture contract — a second-class artifact that undercuts the whole
  reason to publish. `test_inference` would run but return mis-shaped output.

Consequence: coverage is **"any flavor whose serving maps to a Workbench
harness,"** not literally any MLflow model.

### API sketch

A classmethod on the client `Model` (source-less, so it does not hang off
`FeatureSet.to_model`):

```python
model = Model.from_mlflow(
    mlflow_model="runs:/<run_id>/model",   # or a registry uri / local MLmodel dir
    name="aqsol-xgb-local",
    model_type=ModelType.REGRESSOR,
    target_column="solubility",            # not in an MLmodel — caller supplies
    id_column="id",                        # not in an MLmodel — caller supplies
    feature_list=None,                     # default: derive from MLmodel signature
)
```

Then the rest is the **existing** chain — no new code downstream:

```python
model.onboard()          # type/target/features/id, load metrics
end = model.to_endpoint()
end.test_inference()     # passes *because* of the rewrap
```

## Scope

**First:** `xgboost` and `sklearn` flavors — they map cleanly to Workbench's
existing harnesses, so the rewrap is a small lift. These are also the **tabular**
models that are worth iterating locally in the first place (chemprop/pytorch need
a GPU, so local iteration was never their story).

**Later:** `pytorch` / `chemprop` — real per-flavor harness work; add only when
there's demand.

**Out of scope:** the MLflow tracking server, registry sync, and any "run
comparison in Workbench." Workbench already owns the published-artifact + report
story; tracking stays MLflow's job.

## Metadata sourcing (where the MLflow specifics concentrate)

`copy()` inherits onboarding metadata for free from its source model.
`mlflow_to_model` must source it:

| field | source |
|---|---|
| `model_framework` | MLmodel flavor |
| `feature_list` | MLmodel signature input schema (or caller override) |
| raw model object | flavor data (e.g. `xgboost`/`sklearn` flavor) |
| **`target_column`** | **caller (required)** — absent from an MLmodel |
| **`id_column`** | **caller (required)** — absent from an MLmodel |
| `model_type` | caller (required) — REGRESSOR/CLASSIFIER/… |

## Open questions

- **Signature reliability.** How often do locally-logged MLmodels carry a usable
  input signature? If frequently missing, `feature_list` becomes effectively
  required too.
- **Flavor → harness mapping.** Confirm the `xgboost` flavor yields a booster the
  existing xgb harness loads as-is, and the `sklearn` flavor an estimator the
  sklearn harness loads as-is — i.e. the rewrap is pure repackaging, no retrain.
- **`pyfunc`-only models.** Custom models logged as generic `pyfunc` (no typed
  flavor) have no raw estimator to extract — likely unsupported under rewrap;
  decide whether to reject cleanly or fall back.
- **Validation/metrics.** A rewrapped model has no Workbench training job, so
  training metrics don't exist until `test_inference` / `cross_fold_inference`
  run. Confirm `onboard()`'s `_load_training_metrics()` degrades gracefully.
```
