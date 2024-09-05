import logging
import json

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp


class ParameterStore:
    """ParameterStore: Manages SageWorks parameters in AWS Systems Manager Parameter Store.

    Common Usage:
        ```
        param_store = ParameterStore()
        param_store.list()
        param_store.add("key", "value")
        value = param_store.get("key")

        # Can also store lists and dictionaries
        param_store.add("my_data", {"key": "value", "number": 42})
        retrieved_dict = param_store.get("my_data")
        param_store.delete("my_data")
        ```
    """

    def __init__(self):
        """ParameterStore Init Method"""
        self.log = logging.getLogger("sageworks")

        # Initialize a SageWorks Session (to assume the SageWorks ExecutionRole)
        self.boto_session = AWSAccountClamp().boto_session()

        # Create a Systems Manager (SSM) client for Parameter Store operations
        self.ssm_client = self.boto_session.client("ssm")

        # Prefix all parameter names
        self.prefix = "/sageworks/"

    def list(self, sageworks_only=False) -> list:
        """List all parameters in the AWS Parameter Store.

        Args:
            sageworks_only (bool): List only SageWorks parameters (default: False)

        Returns:
            list: A list of parameter names and details.
        """
        try:
            # Just return the names of the parameters
            response = self.ssm_client.describe_parameters()
            return [param["Name"] for param in response["Parameters"]]

        except Exception as e:
            self.log.error(f"Failed to list parameters: {e}")
            return []

    def get(self, name: str, decrypt: bool = True):
        """Retrieve a parameter value from the AWS Parameter Store.

        Args:
            name (str): The name of the parameter to retrieve.
            decrypt (bool): Whether to decrypt secure string parameters.

        Returns:
            str | list | dict: The parameter value.
        """
        try:
            # Add the prefix to the parameter name
            name = self.prefix + name

            # Retrieve the parameter from Parameter Store
            response = self.ssm_client.get_parameter(Name=name, WithDecryption=decrypt)
            value = response["Parameter"]["Value"]

            # Attempt to parse the value back to its original type
            try:
                parsed_value = json.loads(value)
                return parsed_value
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, return the value as is (assumed to be a simple string)
                return value

        except Exception as e:
            self.log.error(f"Failed to get parameter '{name}': {e}")
            return None

    def add(self, name: str, value, overwrite: bool = False):
        """Add or update a parameter in the AWS Parameter Store.

        Args:
            name (str): The name of the parameter.
            value (str | list | dict): The value of the parameter.
            overwrite (bool): Whether to overwrite an existing parameter.
        """
        try:
            # Add the prefix to the parameter name
            name = self.prefix + name

            # Anything that's not a string gets converted to JSON
            if not isinstance(value, str):
                value = json.dumps(value)
                if len(value) > 4096:
                    raise ValueError("Parameter size exceeds 4KB limit for standard parameters.")

            # Add or update the parameter in Parameter Store
            self.ssm_client.put_parameter(Name=name, Value=value, Type="String", Overwrite=overwrite)
            self.log.info(f"Parameter '{name}' added/updated successfully.")

        except Exception as e:
            self.log.error(f"Failed to add/update parameter '{name}': {e}")
            raise

    def delete(self, name: str):
        """Delete a parameter from the AWS Parameter Store.

        Args:
            name (str): The name of the parameter to delete.
        """
        try:
            # Add the prefix to the parameter name
            name = self.prefix + name

            # Delete the parameter from Parameter Store
            self.ssm_client.delete_parameter(Name=name)
            self.log.info(f"Parameter '{name}' deleted successfully.")
        except Exception as e:
            self.log.error(f"Failed to delete parameter '{name}': {e}")


if __name__ == "__main__":
    """Exercise the ParameterStore Class"""

    # Create a ParameterStore manager
    param_store = ParameterStore()

    # List the parameters
    print("Listing Parameters...")
    print(param_store.list())

    # Add a new parameter
    param_store.add("test", "value", overwrite=True)

    # Get the parameter
    print(f"Getting parameter 'test': {param_store.get('test')}")

    # Add a dictionary as a parameter
    sample_dict = {"key": "str_value", "awesome_value": 4.2}
    param_store.add("my_data", sample_dict, overwrite=True)

    # Retrieve the parameter as a dictionary
    retrieved_value = param_store.get("my_data")
    print("Retrieved value:", retrieved_value)

    # Delete the parameters
    param_store.delete("test")
    param_store.delete("my_data")
