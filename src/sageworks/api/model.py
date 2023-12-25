"""Model: SageWorks Model API Class"""

# SageWorks Imports
from sageworks.core.artifacts.model_core import ModelCore, ModelType  # noqa: F401
from sageworks.core.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint
from sageworks.api.endpoint import Endpoint


class Model(ModelCore):
    """Model: SageWorks Model API Class

    **Common Usage**
    ```
        my_features = Model(name)
        my_features.summary()
        my_features.details()
        my_features.to_endpoint()
    ```
    """

    def __init__(self, name):
        """Model Initialization
        Args:
            name (str): The name of the Model
        """
        # Call superclass init
        super().__init__(name)

    def to_endpoint(self, name: str = None, tags: list = None, serverless: bool = True) -> Endpoint:
        """Create an Endpoint from the Model

        Args:
            name (str): Set the name for the endpoint. If not specified, an automatic name will be generated
            tags (list): Set the tags for the endpoint. If not specified automatic tags will be generated.
            serverless (bool): Set the endpoint to be serverless (default: True)

        Returns:
            Endpoint: The Endpoint created from the Model
        """

        # Create the Endpoint Name and Tags
        endpoint_name = self.uuid.replace("-model", "") + "-end" if name is None else name
        tags = [endpoint_name] if tags is None else tags

        # Create an Endpoint from the Model
        model_to_endpoint = ModelToEndpoint(self.uuid, endpoint_name, serverless=serverless)
        model_to_endpoint.set_output_tags(tags)
        model_to_endpoint.transform()

        # Return the Endpoint
        return Endpoint(endpoint_name)


if __name__ == "__main__":
    """Exercise the Model Class"""
    from pprint import pprint

    # Retrieve an existing Data Source
    my_model = Model("test-model")
    pprint(my_model.summary())
    pprint(my_model.details())

    # Create an Endpoint from the Model
    my_endpoint = my_model.to_endpoint()
    pprint(my_endpoint.summary())
