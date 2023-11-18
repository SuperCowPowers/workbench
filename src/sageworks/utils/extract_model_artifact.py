"""ExtractModelArtifact is a utility class that reanimates a model joblib file."""
import tarfile
import tempfile
import joblib
import awswrangler as wr
import os
import glob
import xgboost  # noqa: F401

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp


class ExtractModelArtifact:
    def __init__(self, endpoint_name):
        """ExtractModelArtifact Class
        Args:
            endpoint_name (str): Name of the endpoint to extract the model artifact from
        """
        self.endpoint_name = endpoint_name

        """
        self.model_artifact_uri = model_artifact_uri
        self.local_dir = self.set_local_dir()
        self.artifact_tar_path = self.set_artifact_tar()
        self.joblib_file_path = self.set_joblib_file()
        self.model_artifact = self.set_model_artifact()
        """

    def get_model_artifact(self):
        """Get the model artifact from the endpoint"""
        model_artifact_uri = self.get_model_data_uri()
        return self.download_and_extract_model(model_artifact_uri)

    def get_model_data_uri(self) -> str:
        """Get the model artifact URI (S3 Path) from the endpoint
        Returns:
            str: URI (S3 Path) to the model artifact
        """

        # Initialize SageMaker client
        sagemaker_client = AWSAccountClamp().sagemaker_client()

        # Get the endpoint configuration
        endpoint_desc = sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
        endpoint_config_desc = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )

        # Extract the model name from the endpoint configuration
        # Assuming single model for simplicity; adjust if handling multiple models
        model_name = endpoint_config_desc["ProductionVariants"][0]["ModelName"]

        # Get the model description using the Model ARN
        model_desc = sagemaker_client.describe_model(ModelName=model_name)
        model_package_arn = model_desc["PrimaryContainer"]["ModelPackageName"]
        model_package_desc = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)

        # Now we have the model package description, we can get the model artifact URI
        inference_spec = model_package_desc.get("InferenceSpecification", {})
        containers = inference_spec.get("Containers", [])
        if containers:
            model_data_url = containers[0].get("ModelDataUrl")
            return model_data_url
        else:
            raise ValueError("ModelDataUrl not found in the model package description")

    @staticmethod
    def download_and_extract_model(model_artifact_uri):
        """Download and extract model artifact from S3, then load the model into memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_tar_path = os.path.join(tmpdir, "model.tar.gz")

            # Downloading the model artifact using awswrangler
            wr.s3.download(path=model_artifact_uri, local_file=local_tar_path)

            with tarfile.open(local_tar_path, "r:gz") as tar:
                tar.extractall(path=tmpdir)

            # Find the .joblib file in the extracted directory
            model_files = glob.glob(os.path.join(tmpdir, "*.joblib"))
            if not model_files:
                raise FileNotFoundError("No .joblib file found in the extracted model artifact.")
            model_file_path = model_files[0]

            # Load the model
            model = joblib.load(model_file_path)

        # Return the model after exiting the temporary directory context
        return model


if __name__ == "__main__":
    """Exercise the ExtractModelArtifact class"""

    # Create the Class and test it out
    my_endpoint = "abalone-regression-end"
    ema = ExtractModelArtifact(my_endpoint)
    model_data_uri = ema.get_model_data_uri()
    print(f"Model Data URI: {model_data_uri}")
    my_model = ema.download_and_extract_model(model_data_uri)
    print(my_model.feature_names_in_)

    # Test the higher level method
    my_model = ema.get_model_artifact()
    print(my_model.feature_names_in_)
