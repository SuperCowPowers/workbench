"""ParameterStore: Manages Workbench parameters in a Cloud Based Parameter Store."""

from typing import Union
import logging

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_parameter_store import AWSParameterStore


class ParameterStore(AWSParameterStore):
    """ParameterStore: Manages Workbench parameters in a Cloud Based Parameter Store.

    Common Usage:
        ```python
        params = ParameterStore()

        # List Parameters
        params.list()

        ['/workbench/abalone_info',
         '/workbench/my_data',
         '/workbench/test',
         '/workbench/pipelines/my_pipeline']

        # Add Key
        params.upsert("key", "value")
        value = params.get("key")

        # Add any data (lists, dictionaries, etc..)
        my_data = {"key": "value", "number": 4.2, "list": [1,2,3]}
        params.upsert("my_data", my_data)

        # Retrieve data
        return_value = params.get("my_data")
        pprint(return_value)

        {'key': 'value', 'list': [1, 2, 3], 'number': 4.2}

        # Delete parameters
        param_store.delete("my_data")
        ```
    """

    def __init__(self):
        """ParameterStore Init Method"""
        self.log = logging.getLogger("workbench")

        # Initialize the SuperClass
        super().__init__()

    def list(self, prefix: str = None) -> list:
        """List all parameters in the AWS Parameter Store, optionally filtering by a prefix.

        Args:
            prefix (str, optional): A prefix to filter the parameters by. Defaults to None.

        Returns:
            list: A list of parameter names and details.
        """
        return super().list(prefix=prefix)

    def get(self, name: str, warn: bool = True, decrypt: bool = True) -> Union[str, list, dict, None]:
        """Retrieve a parameter value from the AWS Parameter Store.

        Args:
            name (str): The name of the parameter to retrieve.
            warn (bool): Whether to log a warning if the parameter is not found.
            decrypt (bool): Whether to decrypt secure string parameters.

        Returns:
            Union[str, list, dict, None]: The value of the parameter or None if not found.
        """
        return super().get(name=name, warn=warn, decrypt=decrypt)

    def upsert(self, name: str, value, overwrite: bool = True):
        """Insert or update a parameter in the AWS Parameter Store.

        Args:
            name (str): The name of the parameter.
            value (str | list | dict): The value of the parameter.
            overwrite (bool): Whether to overwrite an existing parameter (default: True)
        """
        super().upsert(name=name, value=value, overwrite=overwrite)

    def delete(self, name: str):
        """Delete a parameter from the AWS Parameter Store.

        Args:
            name (str): The name of the parameter to delete.
        """
        super().delete(name=name)

    def __repr__(self):
        """Return a string representation of the ParameterStore object."""
        return super().__repr__()


if __name__ == "__main__":
    """Exercise the ParameterStore Class"""

    # Create a ParameterStore manager
    param_store = ParameterStore()

    # List the parameters
    print("Listing Parameters...")
    print(param_store.list())

    # Add a new parameter
    param_store.upsert("/workbench/test", "value", overwrite=True)

    # Get the parameter
    print(f"Getting parameter 'test': {param_store.get('/workbench/test')}")

    # Add a dictionary as a parameter
    sample_dict = {"key": "str_value", "awesome_value": 4.2}
    param_store.upsert("/workbench/my_data", sample_dict, overwrite=True)

    # Retrieve the parameter as a dictionary
    retrieved_value = param_store.get("/workbench/my_data")
    print("Retrieved value:", retrieved_value)

    # List the parameters
    print("Listing Parameters...")
    print(param_store.list())

    # List the parameters with a prefix
    print("Listing Parameters with prefix '/workbench':")
    print(param_store.list("/workbench"))

    # Delete the parameters
    param_store.delete("/workbench/test")
    param_store.delete("/workbench/my_data")

    # Out of scope tests
    param_store.upsert("test", "value")
    param_store.delete("test")
