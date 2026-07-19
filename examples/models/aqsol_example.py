"""End-to-end AQSol example: DataSource -> FeatureSet -> Model -> Endpoint."""

from workbench.api import DataSource, Model, ModelType, ModelFramework, PublicData

FEATURE_LIST = [
    "molwt",
    "mollogp",
    "molmr",
    "heavyatomcount",
    "numhacceptors",
    "numhdonors",
    "numheteroatoms",
    "numrotatablebonds",
    "numvalenceelectrons",
    "numaromaticrings",
    "numsaturatedrings",
    "numaliphaticrings",
    "ringcount",
    "tpsa",
    "labuteasa",
    "balabanj",
    "bertzct",
]

if __name__ == "__main__":

    # Pull the public AQSol solubility data
    df = PublicData().get("comp_chem/aqsol/aqsol_public_data")

    # DataSource -> FeatureSet
    DataSource(df, name="aqsol_data_test")
    fs = DataSource("aqsol_data_test").to_features("aqsol_features_test", id_column="id", tags=["aqsol", "test"])
    fs.set_owner("test")

    # FeatureSet -> Model
    model = fs.to_model(
        name="aqsol-regression-test",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="solubility",
        feature_list=FEATURE_LIST,
        description="AQSol Regression Model (test)",
        tags=["aqsol", "regression", "test"],
    )
    model.set_owner("test")

    # Model -> Endpoint
    end = Model("aqsol-regression-test").to_endpoint(tags=["aqsol", "regression", "test"])
    end.set_owner("test")
    end.test_inference()
    end.cross_fold_inference()
