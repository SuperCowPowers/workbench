"""FeaturesToModel: Train/Create a Model from a Feature Set"""

from pathlib import Path
from typing import Union
from sagemaker.core.resources import ModelPackageGroup
from sagemaker.core.shapes.shapes import MetricDefinition
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.core.training.configs import SourceCode, Compute, StoppingCondition
import awswrangler as wr

import time
import uuid

# Local Imports
from workbench.core.transforms.transform import Transform, TransformInput, TransformOutput
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.core.artifacts.model_core import ModelCore, ModelType, ModelFramework, ModelImages
from workbench.core.artifacts.artifact import Artifact
from workbench.model_scripts.script_generation import generate_model_script, fill_template
from workbench.utils.workbench_logging import _suppress_sagemaker_logging


class FeaturesToModel(Transform):
    """FeaturesToModel: Train/Create a Model from a FeatureSet

    Common Usage:
        ```python
        from workbench.core.transforms.features_to_model.features_to_model import FeaturesToModel
        to_model = FeaturesToModel(feature_name, model_name, model_type=ModelType)
        to_model.set_output_tags(["abalone", "public", "whatever"])
        to_model.transform(target_column="class_number_of_rings",
                           feature_list=["my", "best", "features"])
        ```
    """

    def __init__(
        self,
        feature_name: str,
        model_name: str,
        model_type: ModelType,
        model_framework=ModelFramework.XGBOOST,
        model_class=None,
        model_import_str=None,
        custom_script=None,
        custom_args=None,
        training_image="base_training",
        inference_image="base_inference",
        inference_arch="x86_64",
    ):
        """FeaturesToModel Initialization
        Args:
            feature_name (str): Name of the FeatureSet to use as input
            model_name (str): Name of the Model to create as output
            model_type (ModelType): ModelType.REGRESSOR or ModelType.CLASSIFIER, etc.
            model_framework (ModelFramework, optional): The model framework (default ModelFramework.XGBOOST)
            model_class (str, optional): The scikit model (e.g. KNeighborsRegressor) (default None)
            model_import_str (str, optional): The import string for the model (default None)
            custom_script (str, optional): Custom script to use for the model (default None)
            custom_args (dict, optional): Custom arguments to pass to custom model scripts (default None)
            training_image (str, optional): Training image (default "training")
            inference_image (str, optional): Inference image (default "inference")
            inference_arch (str, optional): Inference architecture (default "x86_64")
        """

        # Make sure the model_name is a valid name
        Artifact.is_name_valid(model_name, delimiter="-", lower_case=False)

        # Call superclass init
        super().__init__(feature_name, model_name)

        # Set up all my instance attributes
        self.input_type = TransformInput.FEATURE_SET
        self.output_type = TransformOutput.MODEL
        self.model_type = model_type
        self.model_framework = model_framework
        self.model_class = model_class
        self.model_import_str = model_import_str
        self.custom_script = str(custom_script) if custom_script else None
        self.custom_args = custom_args if custom_args else {}
        self.model_trainer = None
        self.training_job_name = None
        self.model_description = None
        self.model_training_root = f"{self.models_s3_path}/{self.output_name}/training"
        self.model_feature_list = None
        self.target_column = None
        self.class_labels = None
        self.training_image = training_image
        self.inference_image = inference_image
        self.inference_arch = inference_arch

    def transform_impl(
        self,
        target_column: Union[str, list[str]],
        description: str = None,
        feature_list: list = None,
        train_all_data=False,
        **kwargs,
    ):
        """Generic Features to Model: Note you should create a new class and inherit from
        this one to include specific logic for your Feature Set/Model
        Args:
            target_column (str or list[str]): Column name(s) of the target variable(s)
            description (str): Description of the model (optional)
            feature_list (list[str]): A list of columns for the features (default None, will try to guess)
            train_all_data (bool): Train on ALL (100%) of the data (default False)
        """
        # Set our model description
        self.model_description = description if description is not None else f"Model created from {self.input_name}"

        # Get our Feature Set and snapshot the training view immediately
        feature_set = FeatureSetCore(self.input_name)
        short_id = uuid.uuid4().hex[:6]
        self.model_training_view_name = f"{self.output_name.replace('-', '_')}_training_{short_id}".lower()
        self.log.important(f"Creating Model Training View: {self.model_training_view_name}...")
        self._create_model_training_view(feature_set, self.model_training_view_name)

        # Delete the existing model (if it exists)
        self.log.important(f"Trying to delete existing model {self.output_name}...")
        ModelCore.managed_delete(self.output_name)

        # Create S3 training data from the model-owned snapshot (not the shared training view)
        base_table = feature_set.data_source.table
        model_view_table = f"{base_table}___{self.model_training_view_name}"
        s3_training_path = feature_set.create_s3_training_data(source_table=model_view_table)
        self.log.info(f"Created new training data {s3_training_path}...")

        # Report the target column(s)
        self.target_column = target_column
        # Normalize target_column to a list for internal use
        target_list = [target_column] if isinstance(target_column, str) else (target_column or [])
        self.log.info(f"Target column(s): {self.target_column}")

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
            self.log.warning("Guessing at the feature list, HIGHLY RECOMMENDED to specify an explicit feature list!")
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
            ] + target_list
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
        # ChemProp expects target_column as a list; other templates expect a string
        target_for_template = target_list if self.model_framework == ModelFramework.CHEMPROP else self.target_column
        template_params = {
            "model_imports": self.model_import_str,
            "model_type": self.model_type,
            "model_framework": self.model_framework,
            "model_class": self.model_class,
            "target_column": target_for_template,
            "feature_list": self.model_feature_list,
            "compressed_features": feature_set.get_compressed_features(),
            "model_metrics_s3_path": self.model_training_root,
            "train_all_data": train_all_data,
            "id_column": feature_set.id_column,
            "hyperparameters": kwargs.get("hyperparameters", {}),
        }
        # Custom Script
        if self.custom_script:
            script_path = self.custom_script
            if self.custom_script.endswith(".template"):
                # Model Type is an enumerated type, so we need to convert it to a string
                template_params["model_type"] = template_params["model_type"].value

                # Fill in the custom script template with specific parameters (include any custom args)
                template_params.update(self.custom_args)
                script_path = fill_template(self.custom_script, template_params, "generated_model_script.py")
            self.log.info(f"Custom script path: {script_path}")

        # We're using one of the built-in model script templates
        else:
            # Generate our model script
            script_path = generate_model_script(template_params)

        # Metric Definitions for Regression (matches model script output format)
        if self.model_type in [ModelType.REGRESSOR, ModelType.UQ_REGRESSOR, ModelType.ENSEMBLE_REGRESSOR]:
            metric_definitions = [
                {"Name": "rmse", "Regex": r"rmse: ([0-9.]+)"},
                {"Name": "mae", "Regex": r"mae: ([0-9.]+)"},
                {"Name": "medae", "Regex": r"medae: ([0-9.]+)"},
                {"Name": "r2", "Regex": r"r2: ([0-9.-]+)"},
                {"Name": "spearmanr", "Regex": r"spearmanr: ([0-9.-]+)"},
                {"Name": "support", "Regex": r"support: ([0-9]+)"},
            ]

        # Metric Definitions for Classification
        elif self.model_type == ModelType.CLASSIFIER:
            # We need to get creative with the Classification Metrics
            # Note: Classification only supports single target
            class_target = target_list[0] if target_list else self.target_column

            # Grab all the target column class values (class labels)
            table = feature_set.data_source.table
            self.class_labels = feature_set.query(f'select DISTINCT {class_target} FROM "{table}"')[
                class_target
            ].to_list()

            # Sanity check on the targets
            if len(self.class_labels) > 10:
                msg = f"Too many target classes ({len(self.class_labels)}) for classification, aborting!"
                self.log.critical(msg)
                raise ValueError(msg)

            # Dynamically create the metric definitions (per-class precision/recall/f1/support)
            # Note: Confusion matrix metrics are skipped to stay under SageMaker's 40 metric limit
            metrics = ["precision", "recall", "f1", "support"]
            metric_definitions = []
            for t in self.class_labels:
                for m in metrics:
                    metric_definitions.append({"Name": f"Metrics:{t}:{m}", "Regex": f"Metrics:{t}:{m} ([0-9.]+)"})

        # If the model type is UNKNOWN, our metric_definitions will be empty
        else:
            self.log.important(f"ModelType is {self.model_type}, skipping metric_definitions...")
            metric_definitions = []

        # Take the full script path and extract the entry point and source directory
        entry_point = str(Path(script_path).name)
        source_dir = str(Path(script_path).parent)

        # Create a Sagemaker Model with our script
        image = ModelImages.get_image_uri(self.sm_session.boto_region_name, self.training_image)

        # Use user-specified instance or default based on framework
        train_instance_type = kwargs.get("training_instance")
        if train_instance_type:
            self.log.important(f"Using user-specified instance {train_instance_type}")
        elif self.model_framework in [ModelFramework.CHEMPROP, ModelFramework.PYTORCH]:
            train_instance_type = "ml.g6.2xlarge"  # NVIDIA L4 GPU + 8 vCPUs for data loading
            self.log.important(f"Using GPU instance {train_instance_type} for {self.model_framework.value}")
        else:
            train_instance_type = "ml.m5.xlarge"

        # Convert metric definitions to V3 MetricDefinition objects
        v3_metric_definitions = [MetricDefinition(name=m["Name"], regex=m["Regex"]) for m in metric_definitions]

        # Create ModelTrainer (V3 replacement for Estimator)
        # Use command= to run our entrypoint wrapper, which executes the model script
        # and then bundles inference code/metadata into the model artifacts
        self.model_trainer = ModelTrainer(
            training_image=image,
            source_code=SourceCode(
                source_dir=source_dir,
                command=f"python training_harness.py {entry_point}",
            ),
            compute=Compute(instance_type=train_instance_type, instance_count=1),
            stopping_condition=StoppingCondition(max_runtime_in_seconds=6 * 3600),
            base_job_name=self.output_name,
            role=self.workbench_role_arn,
            sagemaker_session=self.sm_session,
        )
        self.model_trainer.with_metric_definitions(v3_metric_definitions)

        # Train the model
        self.log.important(f"Training the Model {self.output_name} with Training Image {image}...")
        input_data = self.model_trainer.create_input_data_channel("train", s3_training_path)
        _suppress_sagemaker_logging()
        self.model_trainer.train(input_data_config=[input_data], wait=True)

        # Capture the actual training job name (ModelTrainer appends a timestamp to base_job_name)
        self.training_job_name = self.model_trainer._latest_training_job.training_job_name

        # Now delete the training data
        self.log.info(f"Deleting training data {s3_training_path}...")
        wr.s3.delete_objects(
            [s3_training_path, s3_training_path.replace(".csv", ".csv.metadata")],
            boto3_session=self.boto3_session,
        )

        # Create Model and officially Register
        self.log.important(f"Creating new model {self.output_name}...")
        self.create_and_register_model(**kwargs)

    def _create_model_training_view(self, feature_set: FeatureSetCore, model_view_name: str):
        """Create a model-owned training view with its own isolated weights table.

        Instead of copying the FeatureSet's training view (which references a shared weights table),
        this snapshots the current training state into a model-scoped weights table and builds
        a self-contained view that won't break if the shared weights table is later deleted.

        Args:
            feature_set (FeatureSetCore): The source FeatureSet
            model_view_name (str): The model training view name (e.g. "my_model_training")
        """
        from workbench.core.views.view_utils import dataframe_to_table

        # Snapshot the current training view state (id, training flag, optional sample_weight)
        training_view = feature_set.view("training")
        tv_columns = training_view.columns
        id_column = feature_set.id_column

        select_cols = [f'"{id_column}"', '"training"']
        has_weights = "sample_weight" in tv_columns
        if has_weights:
            select_cols.append('"sample_weight"')

        col_str = ", ".join(select_cols)
        tv_df = feature_set.data_source.query(f'SELECT {col_str} FROM "{training_view.table}"')

        # Add default sample_weight if the training view didn't have one
        if not has_weights:
            tv_df["sample_weight"] = 1.0

        # Create model-scoped weights table
        base_table = feature_set.data_source.table
        model_weights_table = f"_{base_table}___{model_view_name}_weights"
        self.log.info(f"Creating model weights table: {model_weights_table}")
        dataframe_to_table(feature_set.data_source, tv_df, model_weights_table)

        # Build model training view SQL: JOIN base table with model's own weights table
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        feature_columns = [c for c in feature_set.columns if c not in aws_cols]
        sql_columns = ", ".join([f't."{c}"' for c in feature_columns])

        model_view_table = f"{base_table}___{model_view_name}"
        create_view_sql = f"""
        CREATE OR REPLACE VIEW "{model_view_table}" AS
        SELECT {sql_columns}, w."sample_weight", w."training"
        FROM "{base_table}" t
        INNER JOIN "{model_weights_table}" w ON t."{id_column}" = w."{id_column}"
        """
        feature_set.data_source.execute_statement(create_view_sql)

    def post_transform(self, **kwargs):
        """Post-Transform: Calling onboard() on the Model"""
        self.log.info("Post-Transform: Calling onboard() on the Model...")
        time.sleep(3)  # Give AWS time to complete Model register

        # Store the model metadata information
        output_model = ModelCore(self.output_name)
        output_model._set_model_type(self.model_type)
        output_model._set_model_framework(self.model_framework)
        output_model.upsert_workbench_meta({"workbench_model_features": self.model_feature_list})
        output_model.upsert_workbench_meta({"workbench_model_target": self.target_column})
        output_model.upsert_workbench_meta({"workbench_training_view": self.model_training_view_name})

        # Store the class labels (if they exist)
        if self.class_labels:
            output_model.set_class_labels(self.class_labels)

        # Call the Model onboard method
        output_model.onboard_with_args(self.model_type, self.target_column, self.model_feature_list)

    def create_and_register_model(self, aws_region=None, **kwargs):
        """Create and Register the Model

        Args:
            aws_region (str, optional): AWS Region to use (default None)
            **kwargs (dict): Additional keyword arguments to pass to the model registration
        """
        from sagemaker.core.resources import TrainingJob

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Create model group (if it doesn't already exist)
        try:
            ModelPackageGroup.create(
                model_package_group_name=self.output_name,
                model_package_group_description=self.model_description,
                tags=aws_tags,
                session=self.boto3_session,
            )
        except Exception:
            self.log.info(f"Model Package Group {self.output_name} may already exist, continuing...")

        # Get the model artifacts URL from the completed training job
        training_job = TrainingJob.get(self.training_job_name, session=self.boto3_session)
        model_data_url = training_job.model_artifacts.s3_model_artifacts

        # Get the inference image URI
        image = ModelImages.get_image_uri(
            self.sm_session.boto_region_name, self.inference_image, architecture=self.inference_arch
        )
        self.log.important(f"Registering model {self.output_name} with Inference Image {image}...")

        # FIXME: V3 SDK Bug — ModelPackage.create() does response["ModelPackageName"] at line 25047
        # of resources.py, but the AWS CreateModelPackage API only returns "ModelPackageArn".
        # This causes a KeyError on every versioned model package creation. Using boto3 as workaround.
        # No existing GitHub issue — consider filing on https://github.com/aws/sagemaker-core/issues
        self.sm_client.create_model_package(
            ModelPackageGroupName=self.output_name,
            ModelPackageDescription=self.model_description,
            InferenceSpecification={
                "Containers": [{"Image": image, "ModelDataUrl": model_data_url}],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
            },
            ModelApprovalStatus="Approved",
        )


if __name__ == "__main__":
    """Exercise the FeaturesToModel Class"""

    # Regression Model
    input_name = "abalone_features"
    output_name = "abalone-regression"
    to_model = FeaturesToModel(input_name, output_name, model_type=ModelType.REGRESSOR)
    to_model.set_output_tags(["test"])
    to_model.transform(target_column="class_number_of_rings", description="Test Abalone Regression")

    # Classification Model
    input_name = "wine_features"
    output_name = "wine-classification"
    to_model = FeaturesToModel(input_name, output_name, ModelType.CLASSIFIER)
    to_model.set_output_tags(["wine", "public"])
    to_model.transform(target_column="wine_class", description="Wine Classification")

    # Quantile Regression Model (Abalone)
    input_name = "abalone_features"
    output_name = "abalone-regression-uq"
    to_model = FeaturesToModel(input_name, output_name, ModelType.UQ_REGRESSOR)
    to_model.set_output_tags(["abalone", "uq"])
    to_model.transform(target_column="class_number_of_rings", description="Abalone UQ Regression")

    # Scikit-Learn Kmeans Clustering Model
    input_name = "wine_features"
    output_name = "wine-clusters"
    to_model = FeaturesToModel(
        input_name,
        output_name,
        model_class="KMeans",  # Clustering algorithm
        model_import_str="from sklearn.cluster import KMeans",  # Import statement for KMeans
        model_type=ModelType.CLUSTERER,
    )
    to_model.set_output_tags(["wine", "clustering"])
    to_model.transform(target_column=None, description="Wine Clustering", train_all_data=True)

    # Scikit-Learn HDBSCAN Clustering Model
    input_name = "wine_features"
    output_name = "wine-clusters-hdbscan"
    to_model = FeaturesToModel(
        input_name,
        output_name,
        model_class="HDBSCAN",  # Density-based clustering algorithm
        model_import_str="from sklearn.cluster import HDBSCAN",
        model_type=ModelType.CLUSTERER,
    )
    to_model.set_output_tags(["wine", "density-based clustering"])
    to_model.transform(target_column=None, description="Wine Clustering with HDBSCAN", train_all_data=True)

    # Scikit-Learn 2D Projection Model using UMAP
    input_name = "wine_features"
    output_name = "wine-2d-projection"
    to_model = FeaturesToModel(
        input_name,
        output_name,
        model_class="UMAP",
        model_import_str="from umap import UMAP",
        model_type=ModelType.PROJECTION,
    )
    to_model.set_output_tags(["wine", "2d-projection"])
    to_model.transform(target_column=None, description="Wine 2D Projection", train_all_data=True)

    # Custom Script Models
    scripts_root = Path(__file__).resolve().parents[3] / "model_scripts"
    my_custom_script = scripts_root / "custom_script_example" / "custom_model_script.py"
    input_name = "wine_features"
    output_name = "wine-custom"
    to_model = FeaturesToModel(input_name, output_name, model_type=ModelType.CLASSIFIER, custom_script=my_custom_script)
    to_model.set_output_tags(["wine", "custom"])
    to_model.transform(target_column="wine_class", description="Wine Custom Classification")

    # Molecular Descriptors Model
    scripts_root = Path(__file__).resolve().parents[3] / "model_scripts"
    my_script = scripts_root / "custom_models" / "chem_info" / "molecular_descriptors.py"
    input_name = "aqsol_features"
    output_name = "test-smiles-to-taut-md-stereo"
    to_model = FeaturesToModel(input_name, output_name, model_type=ModelType.TRANSFORMER, custom_script=my_script)
    to_model.set_output_tags(["smiles", "molecular descriptors"])
    to_model.transform(target_column=None, feature_list=["smiles"], description="Smiles to Molecular Descriptors")

    # Molecular Fingerprints Model
    scripts_root = Path(__file__).resolve().parents[3] / "model_scripts"
    my_script = scripts_root / "custom_models" / "chem_info" / "morgan_fingerprints.py"
    input_name = "aqsol_features"
    output_name = "smiles-to-fingerprints-v0"
    to_model = FeaturesToModel(input_name, output_name, model_type=ModelType.TRANSFORMER, custom_script=my_script)
    to_model.set_output_tags(["smiles", "morgan fingerprints"])
    to_model.transform(target_column=None, feature_list=["smiles"], description="Smiles to Morgan Fingerprints")
