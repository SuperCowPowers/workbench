"""ExtractModelArtifact is a utility class that reanimates a model joblib file."""
import tarfile
import joblib
import xgboost
import awswrangler as wr
import os
import shutil

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp


class ExtractModelArtifact:
    def __init__(self, endpoint_name, model_artifact_uri):
        """
        ExtractModelArtifact Class
        """
        self.aws_account_clamp = AWSAccountClamp()
        self.boto_session = self.aws_account_clamp.boto_session()
        self.endpoint_name = endpoint_name
        self.model_artifact_uri = model_artifact_uri
        self.local_dir = self.set_local_dir()
        self.artifact_tar_path = self.set_artifact_tar()
        self.joblib_file_path = self.set_joblib_file()
        self.model_artifact = self.set_model_artifact()

    # Setterinos

    def set_local_dir(self):
        local_dir = os.path.join(os.getcwd(), f"{self.endpoint_name}_{self.model_artifact_uri.split('/')[-3]}")
        return local_dir

    def set_artifact_tar(self):
        local_tar_name = f"{self.endpoint_name}_{self.model_artifact_uri.split('/')[-3]}_model.tar.gz"
        local_tar_path = os.path.join(self.local_dir, local_tar_name)
        if not os.path.exists(self.local_dir):
            _ = os.mkdir(self.local_dir)
            _ = wr.s3.download(path=self.model_artifact_uri, local_file=local_tar_path, boto3_session=self.boto_session)
        else:
            print("Model tar already downloaded!")

        return local_tar_path

    def set_joblib_file(self):
        tar = tarfile.open(self.artifact_tar_path)
        tar.extractall(path=self.local_dir)
        tar.close()

        job_lib_files = [f for f in os.listdir(self.local_dir) if ".joblib" in f]

        if len(job_lib_files) == 0:
            raise Exception("No joblib files found...")
        elif len(job_lib_files) > 1:
            raise Exception("More than one joblib file found...")

        job_lib_file_path = os.path.join(self.local_dir, job_lib_files[0])

        return job_lib_file_path

    def set_model_artifact(self):
        model_artifact = joblib.load(self.joblib_file_path)

        return model_artifact

    # Getterinos

    def get_boto_session(self):
        return self.boto_session

    def get_endpoint(self):
        return self.endpoint

    def get_model_artifact_uri(self):
        return self.model_artifact_uri

    def get_artifact_tar_path(self):
        return self.artifact_tar_path

    def get_joblib_file_path(self):
        return self.joblib_file_path

    def get_model_artifact(self):
        return self.model_artifact

    # Methoditos
    def remove_files(self):
        path = self.local_dir
        if os.path.exists(path):
            shutil.rmtree(path)


if __name__ == "__main__":
    """Exercise the ExtractModelArtifact class"""

    model_artifact_uri = (
        "s3://sagemaker-us-east-1-116553368710/sagemaker-scikit-learn-2023-11-03-15-49-20-424/output/model.tar.gz"
    )
    endpoint_name = "solubility-test-regression-end"

    # Create the Class and query for metrics
    ema = ExtractModelArtifact(endpoint_name, model_artifact_uri)
    art_uri = ema.get_model_artifact_uri()
    art_tar_path = ema.get_artifact_tar_path()
    jb_path = ema.get_joblib_file_path()
    model_artifact = ema.get_model_artifact()

    print(art_uri)
    print(art_tar_path)
    print(jb_path)
    print(model_artifact.feature_names_in_)

    ema.remove_files()
