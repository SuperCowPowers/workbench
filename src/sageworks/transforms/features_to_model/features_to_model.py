"""FeaturesToModel: Train/Create a Model from a Feature Set"""
import os
import json
from pathlib import Path
from sagemaker.sklearn.estimator import SKLearn

# Local Imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.feature_sets.feature_set import FeatureSet
from sageworks.artifacts.models.model import Model


class FeaturesToModel(Transform):
    def __init__(self, input_uuid: str, output_uuid: str):
        """FeaturesToModel: Train/Create a Model from a Feature Set"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.FEATURE_SET
        self.output_type = TransformOutput.MODEL
        self.estimator = None
        self.model_script_dir = None

    def generate_model_script(self, target: str, feature_list: list[str], model_type) -> str:
        """Fill in the model template with specific target and feature_list
           Args:
               target (str): Column name of the target variable
               feature_list (list[str]): A list of columns for the features
               model_type (str): regression or classification
           Returns:
              str: The name of the generated model script
        """

        # FIXME: Revisit all of this since it's a bit wonky
        script_name = 'generated_xgb_model.py'
        dir_path = Path(__file__).parent.absolute()
        self.model_script_dir = os.path.join(dir_path, 'light_model_harness')
        template_path = os.path.join(self.model_script_dir, 'xgb_model.template')
        output_path = os.path.join(self.model_script_dir, script_name)
        with open(template_path, 'r') as fp:
            xgb_template = fp.read()

        # Template replacements
        xgb_script = xgb_template.replace('{{target}}', target)
        feature_list_str = json.dumps(feature_list)
        xgb_script = xgb_script.replace('{{feature_list}}', feature_list_str)
        xgb_script = xgb_script.replace('{{model_type}}', model_type)

        # Now write out the generated model script and return the name
        with open(output_path, 'w') as fp:
            fp.write(xgb_script)
        return script_name

    def transform_impl(self, target, input_feature_list=None, delete_existing=False):
        """Generic Features to Model: Note you should create a new class and inherit from
           this one to include specific logic for your Feature Set/Model"""

        # Get our Feature Set and create an S3 CSV Training dataset
        feature_set = FeatureSet(self.input_uuid)
        s3_training_path = feature_set.create_s3_training_data()

        # Did they specify a feature list?
        if input_feature_list:
            feature_list = input_feature_list

        # Try to figure out features (this is a big guess)
        else:
            self.log.warning('Guessing at the feature list...')
            all_columns = feature_set.column_names()
            filter_list = ['id', 'event_time', 'write_time', 'api_invocation_time', 'is_deleted'] + [target]
            feature_list = [c for c in all_columns if c not in filter_list]

        # Generate our model script
        script_path = self.generate_model_script(target, feature_list, 'regression')

        # Create a Sagemaker Model with our script
        self.estimator = SKLearn(
            entry_point=script_path,
            source_dir=self.model_script_dir,
            role=self.sageworks_role_arn,
            instance_type='ml.m5.large',
            sagemaker_session=self.sm_session,
            framework_version='1.0-1'
        )

        # Train the estimator
        self.estimator.fit({'train': s3_training_path})

        # Do they want to delete any existing models?
        if delete_existing:
            self.log.info('Trying to delete existing model...')
            delete_model = Model(self.output_uuid)
            delete_model.delete()

        # Create Model and officially Register
        self.log.info(f"Creating new model {self.output_uuid}...")
        self.create_and_register_model()

        # Now add our SageWorks meta data

    @staticmethod
    def convert_to_aws_tags(metadata: dict):
        """Convert a dictionary to the AWS tag format (list of dicts)
           [ {Key: key_name, Value: value}, {..}, ...] """
        return [{'Key': key, 'Value': value} for key, value in metadata.items()]

    def create_and_register_model(self):
        """Create and Register the Model"""

        # Set up our metadata storage
        sageworks_meta = {'sageworks_tags': self.output_tags}
        for key, value in self.output_meta.items():
            sageworks_meta[key] = value
        aws_tags = self.convert_to_aws_tags(sageworks_meta)

        # Create model group (if it doesn't already exist)
        self.sm_client.create_model_package_group(ModelPackageGroupName=self.output_uuid,
                                                  ModelPackageGroupDescription="Test Model: Abalone Regression",
                                                  Tags=aws_tags)

        # Register our model
        model = self.estimator.create_model(role=self.sageworks_role_arn)
        model.register(
            model_package_group_name=self.output_uuid,
            framework_version='1.0.1',
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium"],
            transform_instances=["ml.m5.large"],
            approval_status="Approved",
            description="Test Model: Abalone Regression"
        )


# Simple test of the FeaturesToModel functionality
def test():
    """Test the FeaturesToModel Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'abalone_feature_set'
    output_uuid = 'abalone-regression'
    to_model = FeaturesToModel(input_uuid, output_uuid)
    to_model.set_output_tags(['abalone', 'public'])
    to_model.set_output_meta({'sageworks_input': input_uuid})
    to_model.transform(target='class_number_of_rings', delete_existing=True)


if __name__ == "__main__":
    test()
