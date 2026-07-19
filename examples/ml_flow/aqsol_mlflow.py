"""MLflow equivalent of examples/models/aqsol_example.py.

The Workbench version is four statements: DataSource -> FeatureSet -> Model -> Endpoint.
This is the same pipeline built on open-source MLflow, kept deliberately honest:
every stage Workbench performs is either implemented here or called out as missing.

Runs standalone: `python aqsol_mlflow.py`. A tracking server is only needed for the
web UI -- see README.md.
"""

import subprocess
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# SQLite file, not a server URL. Training, logging, and model registration all work
# with no server running; point this at http://127.0.0.1:5000 to use a tracking server.
TRACKING_URI = "sqlite:///mlflow.db"
DATA_URL = "s3://workbench-public-data/comp_chem/aqsol/aqsol_public_data.csv"
LOCAL_CSV = Path("aqsol_public_data.csv")
TARGET = "Solubility"
N_FOLDS = 5

# The AQSol CSV ships 26 columns. Only these 17 are legitimate model inputs.
# ID/Name/InChI/InChIKey/SMILES are identifiers; SD and Ocurrences are metadata
# *about the measurement* and leak the target. MLflow has no notion of column
# roles, so this whitelist is the only thing standing between you and leakage.
FEATURE_LIST = [
    "MolWt",
    "MolLogP",
    "MolMR",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumValenceElectrons",
    "NumAromaticRings",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "RingCount",
    "TPSA",
    "LabuteASA",
    "BalabanJ",
    "BertzCT",
]


class UQEnsemble(mlflow.pyfunc.PythonModel):
    """Cross-fold ensemble returning prediction, spread, and a conformal interval.

    Workbench's ModelType.UQ_REGRESSOR provides this as an enum value. Here it is
    a class you own: the fold models supply epistemic spread, and the conformal
    quantile of out-of-fold residuals supplies a calibrated interval width.
    """

    def __init__(self, models, conformal_q):
        self.models = models
        self.conformal_q = conformal_q

    def predict(self, context, model_input, params=None):
        stack = np.column_stack([m.predict(model_input[FEATURE_LIST]) for m in self.models])
        prediction = stack.mean(axis=1)
        spread = stack.std(axis=1)
        return pd.DataFrame(
            {
                "prediction": prediction,
                "spread": spread,
                "lower": prediction - self.conformal_q,
                "upper": prediction + self.conformal_q,
            }
        )


def fetch_data() -> pd.DataFrame:
    """Replaces PublicData().get(...). No catalog, no versioning, no lineage."""
    if not LOCAL_CSV.exists():
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", DATA_URL, str(LOCAL_CSV)], check=True)
    return pd.read_csv(LOCAL_CSV)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces DataSource -> to_features().

    Workbench profiles the columns, flags outliers, samples intelligently, and
    publishes a queryable FeatureSet with an id_column. This does none of that;
    it selects columns and drops nulls, in-process, with no artifact left behind.
    """
    keep = ["ID"] + FEATURE_LIST + [TARGET]
    return df[keep].dropna(subset=FEATURE_LIST + [TARGET]).reset_index(drop=True)


def train(df: pd.DataFrame) -> None:
    """Replaces fs.to_model(...). Everything below is what that one call hides."""
    train_df, holdout_df = train_test_split(df, test_size=0.2, random_state=42)
    X, y = train_df[FEATURE_LIST], train_df[TARGET]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("aqsol-regression")

    with mlflow.start_run(run_name="aqsol-uq-ensemble"):
        mlflow.log_params({"n_folds": N_FOLDS, "n_features": len(FEATURE_LIST), "model": "xgboost"})

        # Fold loop: train N models, collect out-of-fold predictions for calibration.
        models, oof = [], np.zeros(len(train_df))
        for fold, (tr, va) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=42).split(X)):
            # Imputer lives inside the artifact so serving never sees raw nulls.
            pipe = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("xgb", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05)),
                ]
            )
            pipe.fit(X.iloc[tr], y.iloc[tr])
            oof[va] = pipe.predict(X.iloc[va])
            models.append(pipe)
            mlflow.log_metric("fold_mae", mean_absolute_error(y.iloc[va], oof[va]), step=fold)

        # Conformal calibration: the 90% quantile of absolute OOF residuals is the
        # half-width that gives ~90% empirical coverage on unseen data.
        conformal_q = float(np.quantile(np.abs(y - oof), 0.90))
        mlflow.log_metric("conformal_half_width", conformal_q)
        oof_mae = mean_absolute_error(y, oof)
        mlflow.log_metric("oof_mae", oof_mae)
        mlflow.log_metric("oof_r2", r2_score(y, oof))

        uq_model = UQEnsemble(models, conformal_q)
        holdout_pred = uq_model.predict(None, holdout_df)
        holdout_r2 = r2_score(holdout_df[TARGET], holdout_pred["prediction"])
        mlflow.log_metric("holdout_mae", mean_absolute_error(holdout_df[TARGET], holdout_pred["prediction"]))
        mlflow.log_metric("holdout_r2", holdout_r2)
        coverage = (
            (holdout_df[TARGET].values >= holdout_pred["lower"]) & (holdout_df[TARGET].values <= holdout_pred["upper"])
        ).mean()
        mlflow.log_metric("holdout_coverage_90", coverage)

        # Signature + input_example are what make the artifact servable. Omit them
        # and the model still registers cleanly, then fails at request time.
        example = holdout_df[FEATURE_LIST].head(5)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=uq_model,
            signature=infer_signature(example, uq_model.predict(None, holdout_df.head(5))),
            input_example=example,
            registered_model_name="aqsol-regression-mlflow",
        )
        print(f"oof_mae={oof_mae:.3f}  holdout_r2={holdout_r2:.3f}  coverage@90={coverage:.3f}")


if __name__ == "__main__":
    train(build_features(fetch_data()))

    # Replaces Model(...).to_endpoint(). There is no local equivalent of a managed
    # serverless endpoint; `mlflow models serve` runs a uvicorn process on your
    # laptop. Deploying to SageMaker requires building and pushing your own
    # serving image to ECR -- see README.md.
    print("\nServe locally with:")
    print("  mlflow models serve -m models:/aqsol-regression-mlflow/1 --port 5001 --env-manager local")
