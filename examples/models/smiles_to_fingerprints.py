"""Create a Smiles to Fingerprint Model/Endpoint"""

# Workbench Imports
from workbench.api import DataSource, FeatureSet, ModelType, Endpoint
from workbench.core.transforms.pandas_transforms import PandasToFeatures
from workbench.utils.model_utils import get_custom_script_path

RECREATE_FP_ENDPOINT = False

if __name__ == "__main__":

    # An Transformer Model/Endpoint that computes Fingerprints
    if RECREATE_FP_ENDPOINT:
        tags = ["smiles", "morgan fingerprints"]
        script_path = get_custom_script_path("chem_info", "morgan_fingerprints.py")
        feature_set = FeatureSet("aqsol_features")
        model = feature_set.to_model(
            name="smiles-to-fingerprints-v0",
            model_type=ModelType.TRANSFORMER,
            feature_list=["smiles"],
            description="Smiles to Morgan Fingerprints",
            tags=tags,
            custom_script=script_path,
        )

        # Create the endpoint for the model
        end = model.to_endpoint(tags=tags)
        end.auto_inference()


    # Now we take a DataSource, compute the fingerprints, and create a new Model/Endpoint for solubility prediction
    ds = DataSource("aqsol_data")
    df = ds.pull_dataframe()

    # Run the data through our Smiles to Fingerprints Endpoint
    fp_end = Endpoint("smiles-to-fingerprints-v0")
    df_with_fp = fp_end.inference(df)

    # Create a Feature Set
    to_features = PandasToFeatures("aqsol_fingerprints")
    to_features.set_input(df_with_fp, id_column="id")
    to_features.set_output_tags(["aqsol", "fingerprints"])
    to_features.transform()

    # Set our compressed features for this FeatureSet
    fs = FeatureSet("aqsol_fingerprints")
    fs.set_compressed_features(["fingerprint"])

    # Now create a Model/Endpoint for solubility prediction using the fingerprints
    features = ["fingerprint"]
    model = fs.to_model(
        name="aqsol-fingerprint-reg-v0",
        model_type=ModelType.UQ_REGRESSOR,
        target_column="solubility",
        feature_list=features,
        description="Model for Aqueous Solubility using Morgan Fingerprints",
        tags=["aqsol", "fingerprints", "regression"]
    )
    model.set_owner("test")
    end = model.to_endpoint(tags=["aqsol", "fingerprints", "regression"])
    end.auto_inference()
