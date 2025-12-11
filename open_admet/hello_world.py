# An example script demonstrating how to create multi-task Chemprop model in Workbench (deploys to AWS)
from workbench.api import DataSource, FeatureSet, Model, ModelType, ModelFramework, Endpoint


# This example shows how to train a multi-task model predicting all 9 ADMET endpoints
# from the Open ADMET challenge: logd, ksol, hlm_clint, mlm_clint, caco_2_papp_a_b,
# caco_2_efflux, mppb, mbpb, mgmb
ADMET_TARGETS = ['logd', 'ksol', 'hlm_clint', 'mlm_clint', 'caco_2_papp_a_b', 'caco_2_efflux', 'mppb', 'mbpb', 'mgmb']

# Create a Workbench DataSource
# DataSources are typically hand off points (ETL jobs, manual uploads, databases, etc)
ds = DataSource("/path/to/my/training_data.csv", name="open_admet")

# FeatureSet are 'model ready' datasets, with a large set of compound features (RDKIT/Mordred/fingerprints)
fs = ds.to_features("open_admet", id_column="molecule_name")

# Now we grab the FeatureSet and create our Multi-Task Chemprop Model
# Note: We often create 4 or 5 models from ONE FeatureSet (e.g. single-target, multi-task, classification, etc)
feature_set = FeatureSet("open_admet")
model = feature_set.to_model (
    name="open-admet-chemprop",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    target_column=ADMET_TARGETS,  # Multi-task: list of 9 targets
    feature_list=["smiles"],
    description="Multi-task ChemProp model for 9 ADMET endpoints",
    tags=["chemprop", "open_admet", "multitask"],
)
model.set_owner("Jill")

# Now deploy a production ready AWS Endpoint for our Model
end = model.to_endpoint(tags=["chemprop", "open_admet", "multitask"])
end.set_owner("Jill")
