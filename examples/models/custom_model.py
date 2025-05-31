from workbench.api import Model, ModelType
from workbench.core.transforms.features_to_model.features_to_model import FeaturesToModel
from workbench.utils.model_utils import get_custom_script_path

fs_name = "solubility_featurized_class_0_fs"
fs_name = "solubility_test_features"
# fs_name = "aqsol_features"


# Molecular Descriptors Custom Model
my_script_path = get_custom_script_path("chem_info", "molecular_descriptors.py")

to_model = FeaturesToModel(fs_name, "smiles-to-md-v0", model_type=ModelType.TRANSFORMER, custom_script=my_script_path)
to_model.set_output_tags(["smiles", "molecular descriptors"])
to_model.transform(target_column=None, feature_list=["smiles"], description="Smiles to Molecular Descriptors")

# Deploy an Endpoint for the Model
my_model = Model("smiles-to-md-v0")
endpoint = my_model.to_endpoint(name="smiles-to-md-v0", tags=["smiles", "molecular descriptors"])

# Run auto-inference on the Endpoint
endpoint.auto_inference(capture=True)
