# AQSol on open-source MLflow

The MLflow equivalent of [`examples/models/aqsol_example.py`](../models/aqsol_example.py).

- `aqsol_mlflow.py` — full pipeline on open-source MLflow.
- `aqsol_workbench_holdout.py` — Workbench side of the head-to-head, same 7,985 training
  rows and same 1,997-row holdout.

## Setup

```bash
cd examples/ml_flow
uv venv && source .venv/bin/activate
uv pip install mlflow xgboost pandas scikit-learn uvicorn fastapi shap
python aqsol_mlflow.py
```

No tracking server needed — training, logging, and registration all work against the
local `sqlite:///mlflow.db` backend. Two *optional*, unrelated servers:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001    # web UI
mlflow models serve -m models:/aqsol-regression-mlflow/1 --port 5002 --env-manager local
```

Run both from the directory holding `mlflow.db`, or the UI comes up empty. Avoid port
5000 — macOS ControlCenter (AirPlay) holds it.

The model server has no web UI; `/` returns `{"detail":"Not Found"}`. Only `/ping`,
`/health`, `/invocations`:

```bash
curl -s -X POST http://127.0.0.1:5002/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["MolWt","MolLogP","MolMR","HeavyAtomCount","NumHAcceptors","NumHDonors","NumHeteroatoms","NumRotatableBonds","NumValenceElectrons","NumAromaticRings","NumSaturatedRings","NumAliphaticRings","RingCount","TPSA","LabuteASA","BalabanJ","BertzCT"], "data": [[392.51,3.9581,102.4454,23,0,0,2,17,142,0,0,0,0,0,158.5206,0,210.3773]]}}'
```

Returns `prediction`, `spread`, `lower`, `upper` per row.

## Head-to-head

Both scripts derive the holdout independently from the same public data with the same
`dropna` / `train_test_split(seed=42)` logic — ids match by construction (symmetric
difference 0). `aqsol_workbench_holdout.py` passes them to `to_model(validation_ids=...)`,
so no Workbench fold model trains on them.

Same 1,997 unseen rows, each model scored by its own 5-fold ensemble:

| | mae | r2 |
| --- | --- | --- |
| Workbench | 0.754 | 0.780 |
| MLflow | 0.747 | 0.785 |

A dead heat. Model quality is not the differentiator — the platform layer is.

## Mapping

| Workbench | MLflow |
| --- | --- |
| `PublicData().get(...)` | `aws s3 cp --no-sign-request` + `pd.read_csv` |
| `DataSource(df, name=...)` | *none* — no profiling, outlier flagging, smart sampling |
| `.to_features(id_column="id")` | column whitelist + `dropna`, in-process, no artifact |
| `.to_model(ModelType.UQ_REGRESSOR)` | hand-written K-fold loop, ensemble class, conformal calibration |
| `.to_endpoint()` | `mlflow models serve` on localhost; SageMaker needs your own ECR image |
| `.cross_fold_inference()` | *none* — the fold loop is the substitute |
| `to_model(validation_ids=...)` | manual `train_test_split` before the fold loop |
| `split_strategy="scaffold"` | *none* — `KFold` unless you build grouping yourself |
| `end.inference(capture_name=...)` | *none* — predictions are fire-and-forget |

Reattaching a new model to a published FeatureSet (what `aqsol_workbench_holdout.py` does)
has no MLflow equivalent — every run re-derives features from scratch.

## Scaffold splits

Workbench defaults to Bemis-Murcko scaffold grouping (`GroupKFold`), so no scaffold
appears in both train and validation. `aqsol_mlflow.py` uses `KFold(shuffle=True)`, which
lets near-identical analogs sit in both folds — an easier question and a higher score.

MLflow has no concept of a scaffold split, and nothing warns that a random-split number on
molecular data is optimistic.

## Nothing here is shared

`mlflow.db` and `mlruns/` live on the machine that ran the script; the UI binds to
`127.0.0.1`. No coworker can see any of it, and deleting the directory destroys the model,
metrics, and plots.

There is also no inference store — predictions are CSV blobs attached to a run, not
queryable, not joinable, with no link between a model and the predictions it made.

Two ways to get a shared, VPC-hosted MLflow:

- **Self-host** — ECS/Fargate + RDS Postgres + S3 + ALB. Auth is the catch: open-source
  MLflow ships only experimental basic-auth, no SSO, no per-experiment permissions.
- **SageMaker managed MLflow** — `aws sagemaker create-mlflow-tracking-server`. IAM auth,
  S3 artifacts, same account model Workbench already uses.

## SageMaker deployment wall

`mlflow deployments create -t sagemaker` resolves its serving image via
`ecr_client.describe_repositories(repositoryNames=["mlflow-pyfunc"])` **in your own
account**. There is no public MLflow serving image — you must run
`mlflow sagemaker build-and-push-container` first, needing local Docker plus
`ecr:CreateRepository` and `ecr:PutImage`.

A standard `DataScientist` SSO role is denied `ecr:DescribeRepositories`, so this path
requires escalating to a role that can write to a container registry.

## UI

MLflow's run view is a **run-comparison** tool — built to overlay many runs, so a single
run renders as sparse single-bar charts. Workbench's model view is a **model-diagnostics**
tool: interactive pred-vs-actual, prediction intervals, confidence coloring, molecule
structure on hover.

Classic ML runs live under the **Model training** toggle; MLflow 3.x defaults the
experiment view to GenAI.

- **Model metrics** — logged scalars.
- **Artifacts** — `prediction_plot.png` plus three SHAP plots from `mlflow.evaluate()`.
  All static PNGs.
- **Model registry** — `aqsol-regression-mlflow` v1.

`mlflow.evaluate()` emits SHAP plots only — no pred-vs-actual or residual plot, so
`prediction_plot()` builds that by hand and logs it via `mlflow.log_figure()`.

## Traps

All verified on mlflow 3.14.0 / xgboost 3.3.0.

- **A category column registers fine and is un-servable.** Without `input_example`,
  `log_model` succeeds and schema enforcement then fails at request time with
  `Can not safely convert category to <U0`. Passing `input_example` turns this into a
  loud failure at log time — which is why the script passes one.
- **`mlflow.sklearn.log_model` rejects XGBoost outright**: `Untrusted types found in the
  file: ['xgboost.core.Booster', 'xgboost.sklearn.XGBRegressor']`. Requires
  `skops_trusted_types=[...]`. The pyfunc path this script uses is unaffected.
- **`mlflow.evaluate()`'s `mean_absolute_error` / `r2_score` cover `SHAP_SAMPLE` rows
  (300), not the full holdout** — they look better simply because the sample is small.
  Compare `holdout_mae` / `holdout_r2`. The cap exists because the evaluator needs a
  scalar callable and cannot consume the 4-column UQ output, which forces the
  model-agnostic `PermutationExplainer` (~4 rows/sec, ~8 min for the full holdout).
- **`mlflow models serve` shells out via `bash -c exec uvicorn`**, so activate the venv
  rather than calling `mlflow` by absolute path, or serving dies with `rc 127`.
- **`SD` and `Ocurrences` describe the measurement, not the molecule**, so they do not
  exist for a novel compound at prediction time. Not target leakage — including them
  moves R² by ~0.001 — but they are still not valid model inputs, and MLflow has no
  notion of column roles to stop you.
