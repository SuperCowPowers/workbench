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
from workbench.utils.model_utils import proximity_model_local, uq_model


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
        data_capture: bool = False,
    ) -> Endpoint:
        """Create an Endpoint from the Model.

        Args:
            name (str): Set the name for the endpoint. If not specified, an automatic name will be generated
            tags (list): Set the tags for the endpoint. If not specified automatic tags will be generated.
            serverless (bool): Set the endpoint to be serverless (default: True)
            mem_size (int): The memory size for the Endpoint in MB (default: 2048)
            max_concurrency (int): The maximum concurrency for the Endpoint (default: 5)
            instance (str): The instance type to use for Realtime(serverless=False) Endpoints (default: "ml.t2.medium")
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

    def prox_model(self, filtered: bool = True):
        """Create a local Proximity Model for this Model

        Args:
            filtered: bool, optional): Use filtered training data for the Proximity Model (default: True)

        Returns:
           Proximity: A local Proximity Model
        """
        return proximity_model_local(self, filtered=filtered)

    def uq_model(self, uq_model_name: str = None, train_all_data: bool = False) -> "Model":
        """Create a Uncertainty Quantification Model for this Model

        Args:
            uq_model_name (str, optional): Name of the UQ Model (if not specified, a name will be generated)
            train_all_data (bool, optional): Whether to train the UQ Model on all data (default: False)

        Returns:
            Model: The UQ Model
        """
        if uq_model_name is None:
            uq_model_name = self.model_name + "-uq"
        return uq_model(self, uq_model_name, train_all_data=train_all_data)


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
