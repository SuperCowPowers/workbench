"""Model: Manages AWS Model Package/Group creation and management.

Models are automatically set up and provisioned for deployment into AWS.
Models can be viewed in the AWS Sagemaker interfaces or in the Workbench
Dashboard UI, which provides additional model details and performance metrics
"""

# Workbench Imports
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts.model_core import ModelCore, ModelType, ModelFramework  # noqa: F401
from workbench.core.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint
from workbench.api.endpoint import Endpoint
from workbench.utils.model_utils import (
    proximity_model_local,
    fingerprint_prox_model_local,
    noise_model_local,
    cleanlab_model_local,
)


class Model(ModelCore):
    """Model: Workbench Model API Class.

    Common Usage:
        ```python
        my_model = Model(name)
        my_model.details()
        my_model.to_endpoint()
        ```
    """

    def details(self, **kwargs) -> dict:
        """Retrieve the Model Details.

        Returns:
            dict: A dictionary of details about the Model
        """
        return super().details(**kwargs)

    def to_endpoint(
        self,
        name: str = None,
        tags: list = None,
        serverless: bool = True,
        mem_size: int = 2048,
        max_concurrency: int = 5,
        instance: str = None,
        data_capture: bool = False,
    ) -> Endpoint:
        """Create an Endpoint from the Model.

        Args:
            name (str): Set the name for the endpoint. If not specified, an automatic name will be generated
            tags (list): Set the tags for the endpoint. If not specified automatic tags will be generated.
            serverless (bool): Set the endpoint to be serverless (default: True)
            mem_size (int): The memory size for the Endpoint in MB (default: 2048)
            max_concurrency (int): The maximum concurrency for the Endpoint (default: 5)
            instance (str): The instance type for Realtime Endpoints (default: None = auto-select based on model)
            data_capture (bool): Enable data capture for the Endpoint (default: False)

        Returns:
            Endpoint: The Endpoint created from the Model
        """

        # Ensure the endpoint_name is valid
        if name:
            Artifact.is_name_valid(name, delimiter="-", lower_case=False)

        # If the endpoint_name wasn't given generate it
        else:
            name = self.name.replace("_features", "") + ""
            name = Artifact.generate_valid_name(name, delimiter="-")

        # Create the Endpoint Tags
        tags = [name] if tags is None else tags

        # Create an Endpoint from the Model
        model_to_endpoint = ModelToEndpoint(self.name, name, serverless=serverless, instance=instance)
        model_to_endpoint.set_output_tags(tags)
        model_to_endpoint.transform(
            mem_size=mem_size,
            max_concurrency=max_concurrency,
            data_capture=data_capture,
        )

        # Set the Endpoint Owner and Return the Endpoint
        end = Endpoint(name)
        end.set_owner(self.get_owner())
        return end

    def prox_model(self, include_all_columns: bool = False):
        """Create a local Proximity Model for this Model

        Args:
              include_all_columns (bool): Include all DataFrame columns in results (default: False)

        Returns:
           FeatureSpaceProximity: A local FeatureSpaceProximity Model
        """
        return proximity_model_local(self, include_all_columns=include_all_columns)

    def fp_prox_model(
        self,
        include_all_columns: bool = False,
        radius: int = 2,
        n_bits: int = 1024,
        counts: bool = False,
    ):
        """Create a local Fingerprint Proximity Model for this Model

        Args:
           include_all_columns (bool): Include all DataFrame columns in results (default: False)
           radius (int): Morgan fingerprint radius (default: 2)
           n_bits (int): Number of bits for the fingerprint (default: 1024)
           counts (bool): Use count fingerprints instead of binary (default: False)

        Returns:
           FingerprintProximity: A local FingerprintProximity Model
        """
        return fingerprint_prox_model_local(
            self, include_all_columns=include_all_columns, radius=radius, n_bits=n_bits, counts=counts
        )

    def noise_model(self):
        """Create a local Noise Model for this Model

        Returns:
           NoiseModel: A local Noise Model
        """
        return noise_model_local(self)

    def cleanlab_model(self):
        """Create a CleanLearning model for this Model's training data.

        Returns:
           CleanLearning: A fitted cleanlab model. Use get_label_issues() to get
           a DataFrame with id_column, label_quality, predicted_label, given_label, is_label_issue.
        """
        return cleanlab_model_local(self)


if __name__ == "__main__":
    """Exercise the Model Class"""
    from pprint import pprint

    # Retrieve an existing Data Source
    my_model = Model("abalone-regression")
    pprint(my_model.summary())
    pprint(my_model.details())

    # Create an Endpoint from the Model (commented out for now)
    # my_endpoint = my_model.to_endpoint()
    # pprint(my_endpoint.summary())

    # Create a local Proximity Model for this Model
    prox_model = my_model.prox_model()
    print(prox_model.neighbors(3398))
