"""ExtractModelArtifact is a utility class that recreates a model from a SageMaker endpoint artifact"""

import tarfile
import tempfile
import awswrangler as wr
import os
import glob
import json
import joblib
import logging

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.deprecated_utils import deprecated

log = logging.getLogger("workbench")

# Try importing xgboost, set flag if unavailable
try:
    import xgboost as xgb
    from xgboost import XGBModel

    XGBOOST_AVAILABLE = True
except ImportError:
    log.warning("XGBoost Python module not found! pip install xgboost")
    XGBOOST_AVAILABLE = False


@deprecated(version="0.9")
class ExtractModelArtifact:
    """
    ExtractModelArtifact is a utility class that retrieves and processes model artifacts
    from an Amazon SageMaker endpoint.
    """

    def __init__(self, endpoint_name: str):
        """
        Initialize ExtractModelArtifact.

        Args:
            endpoint_name (str): Name of the endpoint to extract the model artifact from.
        """
        self.endpoint_name = endpoint_name
        self.sagemaker_client = AWSAccountClamp().sagemaker_client()

    def get_model_artifact(self):
        """
        Get the model artifact from the endpoint.

        Returns:
            The extracted model object or None if unavailable.
        """
        model_artifact_uri, _ = self.get_artifact_uris()
        return self.download_and_extract_model(model_artifact_uri)

    def get_artifact_uris(self) -> tuple:
        """
        Retrieve the model artifact URI (S3 Path) and script artifact URI from the endpoint.

        Returns:
            tuple: (Model artifact URI, Script artifact URI)
        """
        endpoint_desc = self.sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
        endpoint_config_desc = self.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )
        model_name = endpoint_config_desc["ProductionVariants"][0]["ModelName"]
        model_desc = self.sagemaker_client.describe_model(ModelName=model_name)

        # Handle different SageMaker model configurations
        if "Containers" in model_desc:  # Real-time model
            model_package_arn = model_desc["Containers"][0].get("ModelPackageName")
        elif "PrimaryContainer" in model_desc:  # Serverless model
            model_package_arn = model_desc["PrimaryContainer"].get("ModelPackageName")
        else:
            model_package_arn = None

        if model_package_arn is None:
            raise ValueError(
                "ModelPackageName not found in the model description. Check if the model is correctly configured."
            )

        model_package_desc = self.sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
        containers = model_package_desc.get("InferenceSpecification", {}).get("Containers", [])

        if not containers:
            raise ValueError("Containers not found in the model package description")

        model_data_uri = containers[0].get("ModelDataUrl")
        script_uri = containers[0].get("Environment", {}).get("SAGEMAKER_SUBMIT_DIRECTORY")

        if not model_data_uri or not script_uri:
            raise ValueError("Required URIs not found in the model package description")

        return model_data_uri, script_uri

    @staticmethod
    def load_from_json(tmpdir):
        """
        Load model from a JSON file in the given directory.

        Args:
            tmpdir (str): Path to the directory containing model files.

        Returns:
            XGBModel or None if the model cannot be loaded.
        """
        if not XGBOOST_AVAILABLE:
            return None

        model_files = glob.glob(os.path.join(tmpdir, "*_model.json"))
        if not model_files:
            return None

        for model_file in model_files:
            with open(model_file, "r") as f:
                model_json = json.load(f)
            model_type = json.loads(model_json.get("learner", {}).get("attributes", {}).get("scikit_learn", "{}")).get(
                "_estimator_type"
            )

            model_object = xgb.XGBClassifier() if model_type == "classifier" else xgb.XGBRegressor()
            model_object.load_model(model_file)

            if isinstance(model_object, XGBModel):
                return model_object
        return None

    @staticmethod
    def load_from_joblib(tmpdir):
        """
        Load model from a Joblib file in the given directory.

        Args:
            tmpdir (str): Path to the directory containing model files.

        Returns:
            XGBModel or None if the model cannot be loaded.
        """
        if not XGBOOST_AVAILABLE:
            return None

        model_files = glob.glob(os.path.join(tmpdir, "*_model.joblib"))
        if not model_files:
            return None

        for model_file in model_files:
            model_object = joblib.load(model_file)
            if isinstance(model_object, XGBModel):
                return model_object
        return None

    def download_and_extract_model(self, model_artifact_uri):
        """
        Download and extract model artifact from S3, then load the model into memory.

        Args:
            model_artifact_uri (str): S3 URI of the model artifact.

        Returns:
            Extracted model object or None if unavailable.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            local_tar_path = os.path.join(tmpdir, "model.tar.gz")
            wr.s3.download(path=model_artifact_uri, local_file=local_tar_path)

            with tarfile.open(local_tar_path, "r:gz") as tar:
                tar.extractall(path=tmpdir)

            model_return = self.load_from_joblib(tmpdir) or self.load_from_json(tmpdir)

        return model_return

    @staticmethod
    def unpack_artifacts(model_uri: str, source_uri: str, output_path: str = None):
        """
        Unpack the model and script artifacts from S3 URIs into local directories.

        Args:
            model_uri (str): S3 URI for the model artifact (model.tar.gz).
            source_uri (str): S3 URI for the script artifact (script.tar.gz).
            output_path (str): Local directory to unpack the artifacts into. Defaults to None.
        """
        model_name = model_uri.split("/")[-3]
        output_path = output_path or f"/tmp/workbench/{model_name}"
        model_dir, script_dir = os.path.join(output_path, "model"), os.path.join(output_path, "script")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(script_dir, exist_ok=True)

        model_tar_path, script_tar_path = os.path.join(output_path, "model.tar.gz"), os.path.join(
            output_path, "script.tar.gz"
        )
        wr.s3.download(model_uri, model_tar_path)
        with tarfile.open(model_tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir)
        wr.s3.download(source_uri, script_tar_path)
        with tarfile.open(script_tar_path, "r:gz") as tar:
            tar.extractall(path=script_dir)

        print(f"Model artifacts unpacked into: {model_dir}")
        print(f"Script artifacts unpacked into: {script_dir}")


if __name__ == "__main__":
    """Exercise the ExtractModelArtifact class"""

    # Create the Class and test it out
    my_endpoint = "abalone-regression"
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
