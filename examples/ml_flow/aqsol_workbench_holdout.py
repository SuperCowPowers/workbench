"""Workbench half of the Workbench-vs-MLflow holdout comparison.

Mirrors examples/models/aqsol_example.py, with one change: the same 1,997 rows that
aqsol_mlflow.py holds out are passed to to_model() as validation_ids, so no Workbench
fold model ever trains on them. Both pipelines then score the identical unseen rows.

The holdout is not exported between the two scripts -- both derive it from the same
public data with the same split logic and seed, so the row sets match by construction.
"""

from sklearn.model_selection import train_test_split

from workbench.api import FeatureSet, Model, ModelType, ModelFramework, PublicData

# The CSV's native (pre-DataSource) column names, used to reproduce the MLflow split.
CSV_FEATURES = [
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
CSV_TARGET = "Solubility"


def holdout_ids(df) -> list:
    """The exact ids aqsol_mlflow.py holds out.

    Must stay byte-identical to build_features() + train_test_split() in that script:
    same column subset, same dropna, same reset_index, same test_size and seed. Any
    drift in row count or row order silently produces a different split.
    """
    prepped = df[["ID"] + CSV_FEATURES + [CSV_TARGET]].dropna(subset=CSV_FEATURES + [CSV_TARGET]).reset_index(drop=True)
    _, holdout = train_test_split(prepped, test_size=0.2, random_state=42)
    return holdout["ID"].tolist()


if __name__ == "__main__":

    df = PublicData().get("comp_chem/aqsol/aqsol_public_data")
    val_ids = holdout_ids(df)
    print(f"holdout: {len(val_ids)} ids, first 3: {val_ids[:3]}")

    # Use the existing FeatureSet
    fs = FeatureSet("aqsol_features_test")

    # FeatureSet -> Model. validation_ids keeps those rows out of every fold's training.
    model = fs.to_model(
        name="aqsol-regression-test-holdout",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="solubility",
        feature_list=[f.lower() for f in CSV_FEATURES],
        validation_ids=val_ids,
        description="AQSol Regression Model (MLflow holdout comparison)",
        tags=["aqsol", "regression", "test"],
    )
    model.set_owner("test")

    # Model -> Endpoint, then score the shared holdout under its own capture name.
    end = Model("aqsol-regression-test-holdout").to_endpoint(tags=["aqsol", "regression", "test"])
    end.set_owner("test")

    eval_df = fs.pull_dataframe().query("id in @val_ids")
    end.inference(eval_df, capture_name="mlflow_holdout", id_column="id")

    print(Model("aqsol-regression-test-holdout").get_inference_metrics("mlflow_holdout"))
