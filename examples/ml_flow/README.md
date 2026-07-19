# AQSol on open-source MLflow

The MLflow equivalent of [`examples/models/aqsol_example.py`](../models/aqsol_example.py), for
grounding the Workbench-vs-Databricks comparison in having actually used the alternative.

## Setup

```bash
cd examples/ml_flow
uv venv && source .venv/bin/activate
uv pip install mlflow xgboost pandas scikit-learn uvicorn fastapi
```

`mlflow models serve` shells out to `uvicorn` via `bash -c exec uvicorn ...`, so the binary
must be on `PATH` — activate the venv rather than invoking `mlflow` by absolute path, or
serving dies with `rc 127`.

## Run

```bash
python aqsol_mlflow.py
```

No tracking server required — training, logging, and model registration all work against
the local `sqlite:///mlflow.db` backend. A server is only needed to browse the web UI:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

UI at http://127.0.0.1:5000. Classic ML runs live under the **Model training** toggle —
MLflow 3.x defaults the experiment view to GenAI (Traces/Judges/Prompts).

## Mapping

| Workbench | MLflow |
| --- | --- |
| `PublicData().get(...)` | `aws s3 cp --no-sign-request` + `pd.read_csv` |
| `DataSource(df, name=...)` | *(no equivalent)* — no profiling, outlier flagging, or smart sampling |
| `.to_features(id_column="id")` | column whitelist + `dropna`, in-process, no artifact |
| `.to_model(ModelType.UQ_REGRESSOR)` | hand-written K-fold loop, ensemble class, conformal calibration |
| `.to_endpoint()` | `mlflow models serve` on localhost; SageMaker needs your own ECR image |
| `.cross_fold_inference()` | *(no equivalent)* — the fold loop above is the substitute |

## The SageMaker wall

`mlflow deployments create -t sagemaker` resolves its serving image by calling
`ecr_client.describe_repositories(repositoryNames=["mlflow-pyfunc"])` **in your own account**.
There is no public MLflow serving image. You must first run
`mlflow sagemaker build-and-push-container`, which needs local Docker plus
`ecr:CreateRepository` and `ecr:PutImage`.

A standard `DataScientist` SSO role is denied `ecr:DescribeRepositories`, so this path
cannot be completed without escalating to a role that can write to a container registry.

## Known traps

- Verified on mlflow 3.14.0 / xgboost 3.3.0: `oof_mae=0.778`, `holdout_r2=0.785`,
  `coverage@90=0.904` (conformal calibration holding at its 90% target).
- pandas `category` dtype does not survive the JSON round-trip. A model trains, logs
  metrics, and registers as v1 while being completely un-servable — MLflow emits only a
  WARNING. Bake preprocessing into a sklearn `Pipeline` inside the artifact.
- MLflow 3.x serializes sklearn via skops, which rejects XGBoost unless you pass
  `skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]`.
- Feature selection is entirely on you. `SD` and `Ocurrences` are metadata *about the
  solubility measurement*; including them silently leaks the target.
