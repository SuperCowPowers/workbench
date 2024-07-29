"""ExtractModelArtifact is a utility class that reanimates a model file."""

import tarfile
import tempfile
import awswrangler as wr
import os
import glob
import xgboost as xgb
from xgboost import XGBModel
import json
import joblib
import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp

log = logging.getLogger("sageworks")


class ExtractModelArtifact:
    def __init__(self, endpoint_name):
        """ExtractModelArtifact Class
        Args:
            endpoint_name (str): Name of the endpoint to extract the model artifact from
        """
        self.endpoint_name = endpoint_name

        # Initialize SageMaker client
        self.sagemaker_client = AWSAccountClamp().sagemaker_client()

    def get_model_artifact(self):
        """Get the model artifact from the endpoint"""
        model_artifact_uri, _ = self.get_artifact_uris()
        return self.download_and_extract_model(model_artifact_uri)

    def get_artifact_uris(self) -> tuple:
        """Get the model artifact URI (S3 Path) and source artifact URI from the endpoint

        Returns:
            tuple: (Model artifact URI, Script artifact URI)
        """

        # Get the endpoint configuration
        endpoint_desc = self.sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
        endpoint_config_desc = self.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )

        # Extract the model name from the endpoint configuration
        # Assuming single model for simplicity; adjust if handling multiple models
        model_name = endpoint_config_desc["ProductionVariants"][0]["ModelName"]

        # Get the model description using the Model ARN
        model_desc = self.sagemaker_client.describe_model(ModelName=model_name)

        # Check if 'Containers' (real-time) or 'PrimaryContainer' (serverless) is used
        if "Containers" in model_desc:
            # Real-time model
            model_package_arn = model_desc["Containers"][0].get("ModelPackageName")
        elif "PrimaryContainer" in model_desc:
            # Serverless model
            model_package_arn = model_desc["PrimaryContainer"].get("ModelPackageName")
        else:
            model_package_arn = None

        # Throw an error if the model package ARN is not found
        if model_package_arn is None:
            raise ValueError("ModelPackageName not found in the model description")

        # Now get the model package description and from that the model artifact URI
        model_package_desc = self.sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
        inference_spec = model_package_desc.get("InferenceSpecification", {})
        containers = inference_spec.get("Containers", [])

        # Do we have containers for the model package?
        if containers:
            # Get the model artifact URI
            model_data_uri = containers[0].get("ModelDataUrl")
            # Assuming 'source.tar.gz' is also located in the same container specification
            script_uri = containers[0]["Environment"].get("SAGEMAKER_SUBMIT_DIRECTORY")

            # Ensure both URLs are found
            if model_data_uri is None:
                raise ValueError("ModelDataUrl not found in the model package description")
            if script_uri is None:
                raise ValueError("SAGEMAKER_SUBMIT_DIRECTORY not found in the model package description")

            return model_data_uri, script_uri
        else:
            raise ValueError("Containers not found in the model package description")

    @staticmethod
    def load_from_json(tmpdir):
        # Find the model file in the extracted directory
        model_files = glob.glob(os.path.join(tmpdir, "*_model.json"))
        if not model_files:
            return None

        # Instantiate return model
        model_return = None

        # Check each model_file for an XGBModel object
        for model_file in model_files:
            # Get json and get model type
            with open(model_file, "rb") as f:
                model_json = json.load(f)
            model_type = json.loads(model_json.get("learner").get("attributes").get("scikit_learn")).get(
                "_estimator_type"
            )

            # Load based on model type
            if model_type == "classifier":
                model_object = xgb.XGBClassifier()
                model_object.load_model(model_file)

            else:
                model_object = xgb.XGBRegressor()
                model_object.load_model(model_file)

            # Check type
            if isinstance(model_object, XGBModel):
                print(f"{model_file} is a model object.")

                # Set return if type check passes
                model_return = model_object
            else:
                print(f"{model_file} is NOT a model object.")

        return model_return

    @staticmethod
    def load_from_joblib(tmpdir):
        # Deprecation Warning

        # Find the model file in the extracted directory
        model_files = glob.glob(os.path.join(tmpdir, "*_model.joblib"))
        if not model_files:
            return None

        # Instantiate return model
        model_return = None

        # Check each model_file for an XGBModel object
        for model_file in model_files:
            # Load the model
            model_object = joblib.load(model_file)

            # Check type
            if isinstance(model_object, XGBModel):
                print(f"{model_file} is a model object.")

                # Set return if type check passes
                model_return = model_object
            else:
                print(f"{model_file} is NOT a model object.")

        # Return the model after exiting the temporary directory context
        return model_return

    def download_and_extract_model(self, model_artifact_uri):
        """Download and extract model artifact from S3, then load the model into memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_tar_path = os.path.join(tmpdir, "model.tar.gz")

            # Downloading the model artifact using awswrangler
            wr.s3.download(path=model_artifact_uri, local_file=local_tar_path)

            with tarfile.open(local_tar_path, "r:gz") as tar:
                tar.extractall(path=tmpdir)

            # Try loading from joblib first
            model_return = self.load_from_joblib(tmpdir)
            if model_return:
                log.warning("Joblib is being deprecated as an XGBoost model format.")
                log.warning(
                    "Please recreate this model using the Sageworks API or the xgb.XGBModel.save_model() method."
                )

            # If no joblib model, load from json
            else:
                model_return = self.load_from_json(tmpdir)

        # Return the model after exiting the temporary directory context
        return model_return

    @staticmethod
    def unpack_artifacts(model_uri, source_uri, output_path=None):
        """
        Unpack the model and script artifacts from S3 URIs into local directories.

        Args:
            model_uri (str): S3 URI for the model artifact (model.tar.gz)
            source_uri (str): S3 URI for the script artifact (script.tar.gz)
            output_path (str): Local directory to unpack the artifacts into. Defaults to None.
        """
        # Extract the model name from the model_uri
        model_name = model_uri.split("/")[-3]

        if output_path is None:
            output_path = f"/tmp/sageworks/{model_name}"

        model_dir = os.path.join(output_path, "model")
        script_dir = os.path.join(output_path, "script")

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(script_dir, exist_ok=True)

        # Download and unpack model artifact
        model_tar_path = os.path.join(output_path, "model.tar.gz")
        wr.s3.download(model_uri, model_tar_path)
        with tarfile.open(model_tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir)

        # Download and unpack script artifact
        script_tar_path = os.path.join(output_path, "script.tar.gz")
        wr.s3.download(source_uri, script_tar_path)
        with tarfile.open(script_tar_path, "r:gz") as tar:
            tar.extractall(path=script_dir)

        print(f"Model artifacts unpacked into: {model_dir}")
        print(f"Script artifacts unpacked into: {script_dir}")


if __name__ == "__main__":
    """Exercise the ExtractModelArtifact class"""

    # Create the Class and test it out
    my_endpoint = "abalone-regression-full-new-end"
    ema = ExtractModelArtifact(my_endpoint)

    # Test the lower level methods
    model_uri, script_uri = ema.get_artifact_uris()
    print(f"Model Data URI: {model_uri}")
    print(f"Script URI: {script_uri}")
    my_model = ema.download_and_extract_model(model_uri)

    # Unpack the artifacts
    ema.unpack_artifacts(model_uri, script_uri)

    # Test the higher level method
    my_model = ema.get_model_artifact()
    print(my_model)
