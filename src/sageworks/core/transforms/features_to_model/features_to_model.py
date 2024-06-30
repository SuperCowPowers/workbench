"""FeaturesToModel: Train/Create a Model from a Feature Set"""

import os
import json
from pathlib import Path
from sagemaker.sklearn.estimator import SKLearn
import awswrangler as wr
from datetime import datetime

# Local Imports
from sageworks.core.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.core.artifacts.model_core import ModelCore, ModelType
from sageworks.core.artifacts.artifact import Artifact


class FeaturesToModel(Transform):
    """FeaturesToModel: Train/Create a Model from a FeatureSet

    Common Usage:
        ```
        to_model = FeaturesToModel(feature_uuid, model_uuid, model_type=ModelType)
        to_model.set_output_tags(["abalone", "public", "whatever"])
        to_model.transform(target_column="class_number_of_rings",
                           input_feature_list=[feature_list])
        ```
    """

    def __init__(self, feature_uuid: str, model_uuid: str, model_type: ModelType):
        """FeaturesToModel Initialization
        Args:
            feature_uuid (str): UUID of the FeatureSet to use as input
            model_uuid (str): UUID of the Model to create as output
            model_type (ModelType): ModelType.REGRESSOR or ModelType.CLASSIFIER, etc.
        """

        # Make sure the model_uuid is a valid name
        Artifact.ensure_valid_name(model_uuid, delimiter="-")

        # Call superclass init
        super().__init__(feature_uuid, model_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.FEATURE_SET
        self.output_type = TransformOutput.MODEL
        self.model_type = model_type
        self.estimator = None
        self.model_script_dir = None
        self.model_description = None
        self.model_training_root = self.models_s3_path + "/training"
        self.model_feature_list = None
        self.target_column = None
        self.class_labels = None

    def generate_model_script(
        self, target_column: str, feature_list: list[str], model_type: ModelType, train_all_data: bool
    ) -> str:
        """Fill in the model template with specific target and feature_list
        Args:
            target_column (str): Column name of the target variable
            feature_list (list[str]): A list of columns for the features
            model_type (ModelType): ModelType.REGRESSOR or ModelType.CLASSIFIER, etc.
            train_all_data (bool): Train on ALL (100%) of the data
        Returns:
           str: The name of the generated model script
        """

        # FIXME: Revisit all of this since it's a bit wonky
        if model_type == ModelType.REGRESSOR or model_type == ModelType.CLASSIFIER:
            script_name = "generated_xgb_model.py"
            dir_path = Path(__file__).parent.absolute()
            self.model_script_dir = os.path.join(dir_path, "light_xgb_model")
            template_path = os.path.join(self.model_script_dir, "xgb_model.template")
            output_path = os.path.join(self.model_script_dir, script_name)
            with open(template_path, "r") as fp:
                xgb_template = fp.read()

            # Template replacements
            aws_script = xgb_template.replace("{{target_column}}", target_column)
            feature_list_str = json.dumps(feature_list)
            aws_script = aws_script.replace("{{feature_list}}", feature_list_str)
            aws_script = aws_script.replace("{{model_type}}", model_type.value)
            metrics_s3_path = f"{self.model_training_root}/{self.output_uuid}"
            aws_script = aws_script.replace("{{model_metrics_s3_path}}", metrics_s3_path)
            aws_script = aws_script.replace("{{train_all_data}}", str(train_all_data))

        elif model_type == ModelType.QUANTILE_REGRESSOR:
            script_name = "generated_quantile_model.py"
            dir_path = Path(__file__).parent.absolute()
            self.model_script_dir = os.path.join(dir_path, "light_quant_regression")
            template_path = os.path.join(self.model_script_dir, "quant_regression.template")
            output_path = os.path.join(self.model_script_dir, script_name)
            with open(template_path, "r") as fp:
                quant_template = fp.read()

            # Template replacements
            aws_script = quant_template.replace("{{target_column}}", target_column)
            feature_list_str = json.dumps(feature_list)
            aws_script = aws_script.replace("{{feature_list}}", feature_list_str)
            metrics_s3_path = f"{self.model_training_root}/{self.output_uuid}"
            aws_script = aws_script.replace("{{model_metrics_s3_path}}", metrics_s3_path)

        # Now write out the generated model script and return the name
        with open(output_path, "w") as fp:
            fp.write(aws_script)
        return script_name

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
        delete_model = ModelCore(self.output_uuid, force_refresh=True)
        delete_model.delete()

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
            all_columns = feature_set.column_names()
            filter_list = [
                "id",
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
        self.model_feature_list = [c for c in feature_list if c not in remove_columns]
        self.log.important(f"Feature List for Modeling: {self.model_feature_list}")

        # Generate our model script
        script_path = self.generate_model_script(
            self.target_column, self.model_feature_list, self.model_type, train_all_data
        )

        # Metric Definitions for Regression and Classification
        if self.model_type == ModelType.REGRESSOR or self.model_type == ModelType.QUANTILE_REGRESSOR:
            metric_definitions = [
                {"Name": "RMSE", "Regex": "RMSE: ([0-9.]+)"},
                {"Name": "MAE", "Regex": "MAE: ([0-9.]+)"},
                {"Name": "R2", "Regex": "R2: ([0-9.]+)"},
                {"Name": "NumRows", "Regex": "NumRows: ([0-9]+)"},
            ]
        else:
            # We need to get creative with the Classification Metrics

            # Grab all the target column class values (class labels)
            table = feature_set.data_source.get_table_name()
            self.class_labels = feature_set.query(f"select DISTINCT {self.target_column} FROM {table}")[
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

        # Create a Sagemaker Model with our script
        self.estimator = SKLearn(
            entry_point=script_path,
            source_dir=self.model_script_dir,
            role=self.sageworks_role_arn,
            instance_type="ml.m5.large",
            sagemaker_session=self.sm_session,
            framework_version="1.2-1",
            metric_definitions=metric_definitions,
        )

        # Training Job Name based on the Model UUID and today's date
        training_date_time_utc = datetime.utcnow().strftime("%Y-%m-%d-%H-%M")
        training_job_name = f"{self.output_uuid}-{training_date_time_utc}"

        # Train the estimator
        self.estimator.fit({"train": s3_training_path}, job_name=training_job_name)

        # Now delete the training data
        self.log.info(f"Deleting training data {s3_training_path}...")
        wr.s3.delete_objects(
            [s3_training_path, s3_training_path.replace(".csv", ".csv.metadata")],
            boto3_session=self.boto_session,
        )

        # Create Model and officially Register
        self.log.important(f"Creating new model {self.output_uuid}...")
        self.create_and_register_model()

    def post_transform(self, **kwargs):
        """Post-Transform: Calling onboard() on the Model"""
        self.log.info("Post-Transform: Calling onboard() on the Model...")

        # Store the model feature_list and target_column in the sageworks_meta
        output_model = ModelCore(self.output_uuid, model_type=self.model_type, force_refresh=True)
        output_model.upsert_sageworks_meta({"sageworks_model_features": self.model_feature_list})
        output_model.upsert_sageworks_meta({"sageworks_model_target": self.target_column})

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
        model = self.estimator.create_model(role=self.sageworks_role_arn)
        model.register(
            model_package_group_name=self.output_uuid,
            framework_version="1.2.1",
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium"],
            transform_instances=["ml.m5.large"],
            approval_status="Approved",
            description=self.model_description,
        )


if __name__ == "__main__":
    """Exercise the FeaturesToModel Class"""

    """
    # Regression Model
    input_uuid = "abalone_features"
    output_uuid = "abalone-regression"
    to_model = FeaturesToModel(input_uuid, output_uuid, ModelType.REGRESSOR)
    to_model.set_output_tags(["abalone", "public"])
    to_model.transform(target_column="class_number_of_rings", description="Abalone Regression")

    # Classification Model
    input_uuid = "wine_features"
    output_uuid = "wine-classification"
    to_model = FeaturesToModel(input_uuid, output_uuid, ModelType.CLASSIFIER)
    to_model.set_output_tags(["wine", "public"])
    to_model.transform(target_column="wine_class", description="Wine Classification")
    """

    # Quantile Regression Model
    input_uuid = "abalone_features"
    output_uuid = "abalone-quantile-reg"
    to_model = FeaturesToModel(input_uuid, output_uuid, ModelType.QUANTILE_REGRESSOR)
    to_model.set_output_tags(["abalone", "quantiles"])
    to_model.transform(target_column="class_number_of_rings", description="Abalone Quantile Regression")
