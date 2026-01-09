"""Create a Smiles to Fingerprint Model/Endpoint"""

# Workbench Imports
from workbench.api import FeatureSet, ModelType
from workbench.utils.model_utils import get_custom_script_path

fs_name = "aqsol_features"

# A 'Model' to Compute Morgan Fingerprints Features
script_path = get_custom_script_path("chem_info", "morgan_fingerprints.py")
feature_set = FeatureSet(fs_name)
model = feature_set.to_model(
    name="smiles-to-fingerprints-v0",
    model_type=ModelType.TRANSFORMER,
    feature_list=["smiles"],
    description="Smiles to Morgan Fingerprints",
    tags=["smiles", "morgan fingerprints"],
    custom_script=script_path,
)

# Create the endpoint for the model
end = model.to_endpoint(tags=["smiles", "morgan fingerprints"])

# Run inference on the endpoint
end.auto_inference()
