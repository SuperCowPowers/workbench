"""Create the SMILES to Standardize + Tautomer + Molecular Descriptors + Stereo Model and Endpoint"""

# Workbench Imports
from workbench.api import FeatureSet, ModelType, Model
from workbench.utils.model_utils import get_custom_script_path

# fs_name = "aqsol_features"
fs_name = "solubility_featurized_class_0_fs"


script_path = get_custom_script_path("chem_info", "molecular_descriptors.py")
feature_set = FeatureSet(fs_name)
tags = ["smiles", "molecular descriptors", "tautomerized", "stereo"]
model = feature_set.to_model(
    name="smiles-to-taut-md-stereo-v1",
    model_type=ModelType.TRANSFORMER,
    feature_list=["smiles"],
    description="Smiles to Molecular Descriptors",
    tags=tags,
    custom_script=script_path,
)

# Create the endpoint for the model
end = model.to_endpoint(tags=tags, mem_size=4096, max_concurrency=10)

# Run inference on the endpoint
end.auto_inference(capture=True)
