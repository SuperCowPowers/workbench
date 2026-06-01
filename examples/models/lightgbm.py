from pprint import pprint

from workbench.api import FeatureSet, ModelFramework, ModelType

feature_set = FeatureSet("aqsol_features")
model = feature_set.to_model(
    name="aqsol-lightgbm-regression",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.LIGHTGBM,
    target_column="solubility",
    feature_list=[
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
    ],
    tags=["lightgbm", "molecular descriptors"],
    hyperparameters={"n_estimators": 200, "learning_rate": 0.05},
)
pprint(model.details())
