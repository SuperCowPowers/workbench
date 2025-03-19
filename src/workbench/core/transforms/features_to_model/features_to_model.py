"""FeaturesToModel: Train/Create a Model from a Feature Set"""

from pathlib import Path
from sagemaker.estimator import Estimator
import awswrangler as wr
from datetime import datetime, timezone
import time

# Local Imports
from workbench.core.transforms.transform import Transform, TransformInput, TransformOutput
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.core.artifacts.model_core import ModelCore, ModelType, ModelImages
from workbench.core.artifacts.artifact import Artifact
from workbench.model_scripts.script_generation import generate_model_script, fill_template
from workbench.utils.model_utils import supported_instance_types


class FeaturesToModel(Transform):
    """FeaturesToModel: Train/Create a Model from a FeatureSet

    Common Usage:
        ```python
        from workbench.core.transforms.features_to_model.features_to_model import FeaturesToModel
        to_model = FeaturesToModel(feature_uuid, model_uuid, model_type=ModelType)
        to_model.set_output_tags(["abalone", "public", "whatever"])
        to_model.transform(target_column="class_number_of_rings",
                           feature_list=["my", "best", "features"])
        ```
    """

    def __init__(
        self,
        feature_uuid: str,
        model_uuid: str,
        model_type: ModelType,
        scikit_model_class=None,
        model_import_str=None,
        custom_script=None,
        inference_arch="x86_64",
    ):
        """FeaturesToModel Initialization
        Args:
            feature_uuid (str): UUID of the FeatureSet to use as input
            model_uuid (str): UUID of the Model to create as output
            model_type (ModelType): ModelType.REGRESSOR or ModelType.CLASSIFIER, etc.
            scikit_model_class (str, optional): The scikit model (e.g. KNeighborsRegressor) (default None)
            model_import_str (str, optional): The import string for the model (default None)
            custom_script (str, optional): Custom script to use for the model (default None)
            inference_arch (str, optional): Inference architecture (default "x86_64")
        """

        # Make sure the model_uuid is a valid name
        Artifact.is_name_valid(model_uuid, delimiter="-", lower_case=False)

        # Call superclass init
        super().__init__(feature_uuid, model_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.FEATURE_SET
        self.output_type = TransformOutput.MODEL
        self.model_type = model_type
        self.scikit_model_class = scikit_model_class
        self.model_import_str = model_import_str
        self.custom_script = str(custom_script) if custom_script else None
        self.estimator = None
        self.model_description = None
        self.model_training_root = self.models_s3_path + "/training"
        self.model_feature_list = None
        self.target_column = None
        self.class_labels = None
        self.inference_arch = inference_arch

    def transform_impl(
        self, target_column: str, description: str = None, feature_list: list = None, train_all_data=False
    ):
        """Generic Features to Model: Note you should create a new class and inherit from
        this one to include specific logic for your Feature Set/Model
        Args:
            target_column (str): Column name of the target variable
            description (str): Description of the model (optional)
            feature_list (list[str]): A list of columns for the features (default None, will try to guess)
            train_all_data (bool): Train on ALL (100%) of the data (default False)
        """
        # Delete the existing model (if it exists)
        self.log.important("Trying to delete existing model...")
        ModelCore.managed_delete(self.output_uuid)

        # Set our model description
        self.model_description = description if description is not None else f"Model created from {self.input_uuid}"

        # Get our Feature Set and create an S3 CSV Training dataset
        feature_set = FeatureSetCore(self.input_uuid)
        s3_training_path = feature_set.create_s3_training_data()
        self.log.info(f"Created new training data {s3_training_path}...")

        # Report the target column
        self.target_column = target_column
        self.log.info(f"Target column: {self.target_column}")

        # Did they specify a feature list?
        if feature_list:
            # AWS Feature Groups will also add these implicit columns, so remove them
            aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time", "training"]
            feature_list = [c for c in feature_list if c not in aws_cols]

        # If they didn't specify a feature list, try to guess it
        else:
            # Try to figure out features with this logic
            # - Don't include id, event_time, __index_level_0__, or training columns
            # - Don't include AWS generated columns (e.g. write_time, api_invocation_time, is_deleted)
            # - Don't include the target columns
            # - Don't include any columns that are of type string or timestamp
            # - The rest of the columns are assumed to be features
            self.log.warning("Guessing at the feature list, HIGHLY SUGGESTED to specify an explicit feature list!")
            all_columns = feature_set.columns
            filter_list = [
                "id",
                "auto_id",
                "__index_level_0__",
                "write_time",
                "api_invocation_time",
                "is_deleted",
                "event_time",
                "training",
            ] + [self.target_column]
            feature_list = [c for c in all_columns if c not in filter_list]

            # AWS Feature Store has 3 user column types (String, Integral, Fractional)
            # and two internal types (Timestamp and Boolean). A Feature List for
            # modeling can only contain Integral and Fractional types.
            remove_columns = []
            column_details = feature_set.column_details()
            for column_name in feature_list:
                if column_details[column_name] not in ["Integral", "Fractional"]:
                    self.log.warning(
                        f"Removing {column_name} from feature list, improper type {column_details[column_name]}"
                    )
                    remove_columns.append(column_name)

            # Remove the columns that are not Integral or Fractional
            feature_list = [c for c in feature_list if c not in remove_columns]

        # Set the final feature list
        self.model_feature_list = feature_list
        self.log.important(f"Feature List for Modeling: {self.model_feature_list}")

        # Set up our parameters for the model script
        template_params = {
            "model_imports": self.model_import_str,
            "model_type": self.model_type,
            "scikit_model_class": self.scikit_model_class,
            "target_column": self.target_column,
            "feature_list": self.model_feature_list,
            "model_metrics_s3_path": f"{self.model_training_root}/{self.output_uuid}",
            "train_all_data": train_all_data,
            "id_column": feature_set.id_column,
        }

        # Custom Script
        if self.custom_script:
            script_path = self.custom_script
            if self.custom_script.endswith(".template"):
                script_path = fill_template(self.custom_script, template_params, "generated_model_script.py")
            self.log.info(f"Custom script path: {script_path}")

        # We're using one of the built-in model script templates
        else:
            # Generate our model script
            script_path = generate_model_script(template_params)

        # Metric Definitions for Regression
        if self.model_type == ModelType.REGRESSOR or self.model_type == ModelType.QUANTILE_REGRESSOR:
            metric_definitions = [
                {"Name": "RMSE", "Regex": "RMSE: ([0-9.]+)"},
                {"Name": "MAE", "Regex": "MAE: ([0-9.]+)"},
                {"Name": "R2", "Regex": "R2: ([0-9.]+)"},
                {"Name": "NumRows", "Regex": "NumRows: ([0-9]+)"},
            ]

        # Metric Definitions for Classification
        elif self.model_type == ModelType.CLASSIFIER:
            # We need to get creative with the Classification Metrics

            # Grab all the target column class values (class labels)
            table = feature_set.data_source.table
            self.class_labels = feature_set.query(f'select DISTINCT {self.target_column} FROM "{table}"')[
                self.target_column
            ].to_list()

            # Sanity check on the targets
            if len(self.class_labels) > 10:
                msg = f"Too many target classes ({len(self.class_labels)}) for classification, aborting!"
                self.log.critical(msg)
                raise ValueError(msg)

            # Dynamically create the metric definitions
            metrics = ["precision", "recall", "fscore"]
            metric_definitions = []
            for t in self.class_labels:
                for m in metrics:
                    metric_definitions.append({"Name": f"Metrics:{t}:{m}", "Regex": f"Metrics:{t}:{m} ([0-9.]+)"})

            # Add the confusion matrix metrics
            for row in self.class_labels:
                for col in self.class_labels:
                    metric_definitions.append(
                        {"Name": f"ConfusionMatrix:{row}:{col}", "Regex": f"ConfusionMatrix:{row}:{col} ([0-9.]+)"}
                    )

        # If the model type is UNKNOWN, our metric_definitions will be empty
        else:
            self.log.important(f"ModelType is {self.model_type}, skipping metric_definitions...")
            metric_definitions = []

        # Take the full script path and extract the entry point and source directory
        entry_point = str(Path(script_path).name)
        source_dir = str(Path(script_path).parent)

        # Create a Sagemaker Model with our script
        image = ModelImages.get_image_uri(self.sm_session.boto_region_name, "training", "0.1")
        self.estimator = Estimator(
            entry_point=entry_point,
            source_dir=source_dir,
            role=self.workbench_role_arn,
            instance_count=1,
            instance_type="ml.m5.large",
            sagemaker_session=self.sm_session,
            image_uri=image,
            metric_definitions=metric_definitions,
        )

        # Training Job Name based on the Model UUID and today's date
        training_date_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")
        training_job_name = f"{self.output_uuid}-{training_date_time_utc}"

        # Train the estimator
        self.estimator.fit({"train": s3_training_path}, job_name=training_job_name)

        # Now delete the training data
        self.log.info(f"Deleting training data {s3_training_path}...")
        wr.s3.delete_objects(
            [s3_training_path, s3_training_path.replace(".csv", ".csv.metadata")],
            boto3_session=self.boto3_session,
        )

        # Create Model and officially Register
        self.log.important(f"Creating new model {self.output_uuid}...")
        self.create_and_register_model()

    def post_transform(self, **kwargs):
        """Post-Transform: Calling onboard() on the Model"""
        self.log.info("Post-Transform: Calling onboard() on the Model...")
        time.sleep(3)  # Give AWS time to complete Model register

        # Store the model feature_list and target_column in the workbench_meta
        output_model = ModelCore(self.output_uuid, model_type=self.model_type)
        output_model.upsert_workbench_meta({"workbench_model_features": self.model_feature_list})
        output_model.upsert_workbench_meta({"workbench_model_target": self.target_column})

        # Store the class labels (if they exist)
        if self.class_labels:
            output_model.set_class_labels(self.class_labels)

        # Call the Model onboard method
        output_model.onboard_with_args(self.model_type, self.target_column, self.model_feature_list)

    def create_and_register_model(self):
        """Create and Register the Model"""

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Create model group (if it doesn't already exist)
        self.sm_client.create_model_package_group(
            ModelPackageGroupName=self.output_uuid,
            ModelPackageGroupDescription=self.model_description,
            Tags=aws_tags,
        )

        # Register our model
        image = ModelImages.get_image_uri(self.sm_session.boto_region_name, "inference", "0.1", self.inference_arch)
        self.log.important(f"Registering model {self.output_uuid} with image {image}...")
        model = self.estimator.create_model(role=self.workbench_role_arn)
        model.register(
            model_package_group_name=self.output_uuid,
            image_uri=image,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=supported_instance_types(self.inference_arch),
            transform_instances=["ml.m5.large", "ml.m5.xlarge"],
            approval_status="Approved",
            description=self.model_description,
        )


if __name__ == "__main__":
    """Exercise the FeaturesToModel Class"""

    # Regression Model
    input_uuid = "abalone_features"
    output_uuid = "abalone-regression"
    to_model = FeaturesToModel(input_uuid, output_uuid, model_type=ModelType.REGRESSOR)
    to_model.set_output_tags(["abalone", "public"])
    to_model.transform(target_column="class_number_of_rings", description="Abalone Regression")

    """
    # Classification Model
    input_uuid = "wine_features"
    output_uuid = "wine-classification"
    to_model = FeaturesToModel(input_uuid, output_uuid, ModelType.CLASSIFIER)
    to_model.set_output_tags(["wine", "public"])
    to_model.transform(target_column="wine_class", description="Wine Classification")

    # Quantile Regression Model (Abalone)
    input_uuid = "abalone_features"
    output_uuid = "abalone-quantile-reg"
    to_model = FeaturesToModel(input_uuid, output_uuid, ModelType.QUANTILE_REGRESSOR)
    to_model.set_output_tags(["abalone", "quantiles"])
    to_model.transform(target_column="class_number_of_rings", description="Abalone Quantile Regression")

    # Scikit-Learn Kmeans Clustering Model
    input_uuid = "wine_features"
    output_uuid = "wine-clusters"
    to_model = FeaturesToModel(
        input_uuid,
        output_uuid,
        scikit_model_class="KMeans",  # Clustering algorithm
        model_import_str="from sklearn.cluster import KMeans",  # Import statement for KMeans
        model_type=ModelType.CLUSTERER,
    )
    to_model.set_output_tags(["wine", "clustering"])
    to_model.transform(target_column=None, description="Wine Clustering", train_all_data=True)

    # Scikit-Learn HDBSCAN Clustering Model
    input_uuid = "wine_features"
    output_uuid = "wine-clusters-hdbscan"
    to_model = FeaturesToModel(
        input_uuid,
        output_uuid,
        scikit_model_class="HDBSCAN",  # Density-based clustering algorithm
        model_import_str="from sklearn.cluster import HDBSCAN",
        model_type=ModelType.CLUSTERER,
    )
    to_model.set_output_tags(["wine", "density-based clustering"])
    to_model.transform(target_column=None, description="Wine Clustering with HDBSCAN", train_all_data=True)

    # Scikit-Learn 2D Projection Model using UMAP
    input_uuid = "wine_features"
    output_uuid = "wine-2d-projection"
    to_model = FeaturesToModel(
        input_uuid,
        output_uuid,
        scikit_model_class="UMAP",
        model_import_str="from umap import UMAP",
        model_type=ModelType.PROJECTION,
    )
    to_model.set_output_tags(["wine", "2d-projection"])
    to_model.transform(target_column=None, description="Wine 2D Projection", train_all_data=True)

    # Custom Script Models
    scripts_root = Path(__file__).resolve().parents[3] / "model_scripts"
    my_custom_script = scripts_root / "custom_script_example" / "custom_model_script.py"
    input_uuid = "wine_features"
    output_uuid = "wine-custom"
    to_model = FeaturesToModel(input_uuid, output_uuid, model_type=ModelType.CLASSIFIER, custom_script=my_custom_script)
    to_model.set_output_tags(["wine", "custom"])
    to_model.transform(target_column="wine_class", description="Wine Custom Classification")

    # Molecular Descriptors Model
    scripts_root = Path(__file__).resolve().parents[3] / "model_scripts"
    my_script = scripts_root / "custom_models" / "chem_info" / "molecular_descriptors.py"
    input_uuid = "aqsol_features"
    output_uuid = "smiles-to-md-v0"
    to_model = FeaturesToModel(input_uuid, output_uuid, model_type=ModelType.TRANSFORMER, custom_script=my_script)
    to_model.set_output_tags(["smiles", "molecular descriptors"])
    to_model.transform(target_column=None, feature_list=["smiles"], description="Smiles to Molecular Descriptors")

    # Molecular Fingerprints Model
    scripts_root = Path(__file__).resolve().parents[3] / "model_scripts"
    my_script = scripts_root / "custom_models" / "chem_info" / "morgan_fingerprints.py"
    input_uuid = "aqsol_features"
    output_uuid = "smiles-to-fingerprints-v0"
    to_model = FeaturesToModel(input_uuid, output_uuid, model_type=ModelType.TRANSFORMER, custom_script=my_script)
    to_model.set_output_tags(["smiles", "morgan fingerprints"])
    to_model.transform(target_column=None, feature_list=["smiles"], description="Smiles to Morgan Fingerprints")

    # Tautomerization Model
    scripts_root = Path(__file__).resolve().parents[3] / "model_scripts"
    my_script = scripts_root / "custom_models" / "chem_info" / "tautomerize.py"
    input_uuid = "aqsol_features"
    output_uuid = "tautomerize-v0"
    to_model = FeaturesToModel(input_uuid, output_uuid, model_type=ModelType.TRANSFORMER, custom_script=my_script)
    to_model.set_output_tags(["smiles", "tautomerization"])
    to_model.transform(target_column=None, feature_list=["smiles"], description="Tautomerize Smiles")
    """
