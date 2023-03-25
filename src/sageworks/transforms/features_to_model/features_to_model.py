"""FeaturesToModel: Train/Create a Model from a Feature Set"""
import os
import json
from pathlib import Path
from sagemaker.sklearn.estimator import SKLearn

# Local Imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.feature_sets.feature_set import FeatureSet


class FeaturesToModel(Transform):
    def __init__(self, input_uuid: str = None, output_uuid: str = None):
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

    def transform_impl(self, **kwargs):
        """Compute a Feature Set based on RDKit Descriptors"""

        # Get our Feature Set and create an S3 CSV Training dataset
        feature_set = FeatureSet(self.input_uuid)
        s3_training_path = feature_set.create_s3_training_data()

        # Figure out features (FIXME)
        all_columns = feature_set.column_names()
        filter_list = ['id', 'smiles', 'eventtime', 'solubility', 'write_time', 'api_invocation_time', 'is_deleted']
        feature_columns = [c for c in all_columns if c not in filter_list]

        # Generate our model script
        script_path = self.generate_model_script('solubility', feature_columns, 'regression')

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

        # Create Model and officially Register
        self.create_and_register_model()

    def create_and_register_model(self):
        """Create and Register the Model"""

        # Set up our information and tags
        specs = {
            "input": self.input_uuid,
            "output": self.output_uuid,
            "info": "Test Model: Solubility Regression",
            "tags": ['sageworks']
        }
        model_specs = json.dumps(specs)
        model = self.estimator.create_model(role=self.sageworks_role_arn)
        model.register(
            model_package_group_name='solubility-regression',
            framework_version='1.0.1',
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium"],
            transform_instances=["ml.m5.large"],
            approval_status="Approved",
            description=model_specs
        )


# Simple test of the FeaturesToModel functionality
def test():
    """Test the FeaturesToModel Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'test_rdkit_features'
    output_uuid = 'test_solubility_regression'
    FeaturesToModel(input_uuid, output_uuid).transform(delete_existing=True)


if __name__ == "__main__":
    test()
