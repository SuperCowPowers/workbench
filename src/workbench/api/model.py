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

    def to_endpoint(self, name: str = None, tags: list = None, serverless: bool = True) -> Endpoint:
        """Create an Endpoint from the Model.

        Args:
            name (str): Set the name for the endpoint. If not specified, an automatic name will be generated
            tags (list): Set the tags for the endpoint. If not specified automatic tags will be generated.
            serverless (bool): Set the endpoint to be serverless (default: True)

        Returns:
            Endpoint: The Endpoint created from the Model
        """

        # Ensure the endpoint_name is valid
        if name:
            Artifact.is_name_valid(name, delimiter="-", lower_case=False)

        # If the endpoint_name wasn't given generate it
        else:
            name = self.uuid.replace("_features", "") + "-end"
            name = Artifact.generate_valid_name(name, delimiter="-")

        # Create the Endpoint Tags
        tags = [name] if tags is None else tags

        # Create an Endpoint from the Model
        model_to_endpoint = ModelToEndpoint(self.uuid, name, serverless=serverless)
        model_to_endpoint.set_output_tags(tags)
        model_to_endpoint.transform()

        # Return the Endpoint
        return Endpoint(name)


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
