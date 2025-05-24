"""Model: Manages AWS Model Package/Group creation and management.

Models are automatically set up and provisioned for deployment into AWS.
Models can be viewed in the AWS Sagemaker interfaces or in the Workbench
Dashboard UI, which provides additional model details and performance metrics
"""

# Workbench Imports
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts.model_core import ModelCore, ModelType  # noqa: F401
from workbench.core.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint
from workbench.api.endpoint import Endpoint
from workbench.utils.model_utils import proximity_model


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
        instance: str = "ml.t2.medium",
    ) -> Endpoint:
        """Create an Endpoint from the Model.

        Args:
            name (str): Set the name for the endpoint. If not specified, an automatic name will be generated
            tags (list): Set the tags for the endpoint. If not specified automatic tags will be generated.
            serverless (bool): Set the endpoint to be serverless (default: True)
            mem_size (int): The memory size for the Endpoint in MB (default: 2048)
            max_concurrency (int): The maximum concurrency for the Endpoint (default: 5)
            instance (str): The instance type to use for Realtime(serverless=False) Endpoints (default: "ml.t2.medium")

        Returns:
            Endpoint: The Endpoint created from the Model
        """

        # Ensure the endpoint_name is valid
        if name:
            Artifact.is_name_valid(name, delimiter="-", lower_case=False)

        # If the endpoint_name wasn't given generate it
        else:
            name = self.uuid.replace("_features", "") + ""
            name = Artifact.generate_valid_name(name, delimiter="-")

        # Create the Endpoint Tags
        tags = [name] if tags is None else tags

        # Create an Endpoint from the Model
        model_to_endpoint = ModelToEndpoint(self.uuid, name, serverless=serverless, instance=instance)
        model_to_endpoint.set_output_tags(tags)
        model_to_endpoint.transform(
            mem_size=mem_size,
            max_concurrency=max_concurrency,
        )

        # Set the Endpoint Owner and Return the Endpoint
        end = Endpoint(name)
        end.set_owner(self.get_owner())
        return end

    def prox_model(self, prox_model_name: str = None, track_columns: list = None) -> "Model":
        """Create a Proximity Model for this Model

        Args:
            prox_model_name (str, optional): Name of the Proximity Model (if not specified, a name will be generated)
            track_columns (list, optional): List of columns to track in the Proximity Model.

        Returns:
            Model: The Proximity Model
        """
        if prox_model_name is None:
            prox_model_name = self.model_name + "-prox"
        return proximity_model(self, prox_model_name, track_columns=track_columns)


if __name__ == "__main__":
    """Exercise the Model Class"""
    from pprint import pprint

    # Retrieve an existing Data Source
    my_model = Model("abalone-regression")
    pprint(my_model.summary())
    pprint(my_model.details())

    # Create an Endpoint from the Model
    my_endpoint = my_model.to_endpoint()
    pprint(my_endpoint.summary())
