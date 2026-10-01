"""TabICL trained and served on this machine: no AWS account, no SageMaker images.

Needs the modeling extra: pip install "workbench[modeling]"
"""

from workbench.api import ModelFramework, ModelType, PublicData
from workbench.local import DataSource

feature_list = [
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

# Public AqSol data -> local DataSource -> local FeatureSet (column names are stored lowercase)
df = PublicData().get("comp_chem/aqsol/aqsol_public_data")
ds = DataSource(df, name="aqsol_local")
fs = ds.to_features("aqsol_local_features", id_column="ID")

# TabICL Regression Model: fold models give the cross-fold metrics, one cached model serves
model = fs.to_model(
    "aqsol-local-reg-tabicl",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.TABICL,
    target_column="solubility",
    feature_list=feature_list,
)
metrics = model.get_inference_metrics()
print(metrics)

# A local Endpoint loads the model in this process
end = model.to_endpoint()
preds = end.inference(fs.pull_dataframe().head(20))
print(preds[["id", "solubility", "prediction", "prediction_std", "confidence"]].head())
