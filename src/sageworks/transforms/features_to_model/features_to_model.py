"""FeaturesToModel: Train/Create a Model from a Feature Set"""
import os
import json
from pathlib import Path
from sagemaker.sklearn.estimator import SKLearn
import awswrangler as wr

# Local Imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.feature_sets.feature_set import FeatureSet
from sageworks.artifacts.models.model import Model


class FeaturesToModel(Transform):
    """FeaturesToModel: Train/Create a Model from a FeatureSet

    Common Usage:
        to_model = FeaturesToModel(feature_uuid, model_uuid)
        to_model.set_output_tags(["abalone", "public", "whatever"])
        to_model.transform(target="class_number_of_rings", description="Abalone Regression Model".
                           input_feature_list=<features>, model_type="regressor/classifier",
                           )
    """

    def __init__(self, feature_uuid: str, model_uuid: str):
        """FeaturesToModel Initialization"""

        # Call superclass init
        super().__init__(feature_uuid, model_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.FEATURE_SET
        self.output_type = TransformOutput.MODEL
        self.estimator = None
        self.model_script_dir = None
        self.model_description = None

    def generate_model_script(self, target: str, feature_list: list[str], model_type: str) -> str:
        """Fill in the model template with specific target and feature_list
        Args:
            target (str): Column name of the target variable
            feature_list (list[str]): A list of columns for the features
            model_type (str): regressor or classifier
        Returns:
           str: The name of the generated model script
        """

        # FIXME: Revisit all of this since it's a bit wonky
        script_name = "generated_xgb_model.py"
        dir_path = Path(__file__).parent.absolute()
        self.model_script_dir = os.path.join(dir_path, "light_model_harness")
        template_path = os.path.join(self.model_script_dir, "xgb_model.template")
        output_path = os.path.join(self.model_script_dir, script_name)
        with open(template_path, "r") as fp:
            xgb_template = fp.read()

        # Template replacements
        xgb_script = xgb_template.replace("{{target}}", target)
        feature_list_str = json.dumps(feature_list)
        xgb_script = xgb_script.replace("{{feature_list}}", feature_list_str)
        xgb_script = xgb_script.replace("{{model_type}}", model_type)

        # Now write out the generated model script and return the name
        with open(output_path, "w") as fp:
            fp.write(xgb_script)
        return script_name

    def transform_impl(self, target, description, feature_list=None, model_type="regressor"):
        """Generic Features to Model: Note you should create a new class and inherit from
        this one to include specific logic for your Feature Set/Model
        Args:
            target (str): Column name of the target variable
            description (str): Description of the model
            feature_list (list[str]): A list of columns for the features (default None, will try to guess)
            model_type (str): regressor or classifier (default = regressor)
        """
        # Set our model description
        self.model_description = description

        # Get our Feature Set and create an S3 CSV Training dataset
        feature_set = FeatureSet(self.input_uuid)
        s3_training_path = feature_set.create_s3_training_data()
        self.log.info(f"Created new training data {s3_training_path}...")

        # If they didn't specify a feature list, try to guess it
        if feature_list is None:
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
                "event_time",
                "__index_level_0__",
                "training",
                "write_time",
                "api_invocation_time",
                "is_deleted",
            ] + [target]
            feature_list = [c for c in all_columns if c not in filter_list]

            # Now remove any string or datetime columns from the Feature List
            # Note: AWS Feature Store has 4 column types (string, integral, fractional, and timestamp)
            # Note2: The 'bigint' type = integral and 'double' type = fractional
            column_details = feature_set.column_details()
            feature_list = [c for c in feature_list if column_details[c] not in ["string", "timestamp"]]
            self.log.info(f"Guessed feature list: {feature_list}")

        # Generate our model script
        script_path = self.generate_model_script(target, feature_list, model_type)

        # Metric Definitions for Regression and Classification
        if model_type == "regressor":
            metric_definitions = [
                {"Name": "RMSE", "Regex": "RMSE: ([0-9.]+)"},
                {"Name": "MAE", "Regex": "MAE: ([0-9.]+)"},
                {"Name": "R2", "Regex": "R2 Score: ([0-9.]+)"},
            ]
        else:
            # We need to get creative with the Classification Metrics
            # Grab all the target class values
            table = feature_set.data_source.table_name
            targets = feature_set.query(f"select DISTINCT {target} FROM {table}")[target].to_list()
            metrics = ["precision", "recall", "fscore"]

            # Dynamically create the metric definitions
            metric_definitions = []
            for t in targets:
                for m in metrics:
                    metric_definitions.append({"Name": f"Metrics:{t}:{m}", "Regex": f"Metrics:{t}:{m} ([0-9.]+)"})

            # Add the confusion matrix metrics
            for row in targets:
                for col in targets:
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

        # Train the estimator
        self.estimator.fit({"train": s3_training_path})

        # Now delete the training data
        self.log.info(f"Deleting training data {s3_training_path}...")
        wr.s3.delete_objects(
            [s3_training_path, s3_training_path.replace(".csv", ".csv.metadata")],
            boto3_session=self.boto_session,
        )

        # Delete the existing model (if it exists)
        self.log.info("Trying to delete existing model...")
        delete_model = Model(self.output_uuid)
        delete_model.delete()

        # Create Model and officially Register
        self.log.info(f"Creating new model {self.output_uuid}...")
        self.create_and_register_model()

    def post_transform(self, **kwargs):
        """Post-Transform: Calling make_ready() on the Model"""
        self.log.info("Post-Transform: Calling make_ready() on the Model...")

        # Okay, lets get our output model and set it to initializing
        output_model = Model(self.output_uuid, force_refresh=True)
        output_model.set_status("initializing")

        # Call the Model make_ready method and set status to ready
        output_model.make_ready()

    def create_and_register_model(self):
        """Create and Register the Model"""

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Create model group (if it doesn't already exist)
        self.sm_client.create_model_package_group(
            ModelPackageGroupName=self.output_uuid,
            ModelPackageGroupDescription="Test Model: AQSol Regression",
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

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "abalone_feature_set"
    output_uuid = "abalone-regression"
    to_model = FeaturesToModel(input_uuid, output_uuid)
    to_model.set_output_tags(["abalone", "public"])
    to_model.transform(description="Abalone Regression Model", target="class_number_of_rings")
