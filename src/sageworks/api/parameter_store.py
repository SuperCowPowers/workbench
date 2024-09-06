"""ParameterStore: Manages SageWorks parameters in AWS Systems Manager Parameter Store."""

from typing import Union
import logging
import json
import zlib
import base64

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

    def __init__(self, prefix: Union[str, None] = "/sageworks"):
        """ParameterStore Init Method

        Args:
            prefix (str): The prefix for all parameter names (default: "/sageworks")
        """
        self.log = logging.getLogger("sageworks")

        # Initialize a SageWorks Session (to assume the SageWorks ExecutionRole)
        self.boto_session = AWSAccountClamp().boto_session()

        # Create a Systems Manager (SSM) client for Parameter Store operations
        self.ssm_client = self.boto_session.client("ssm")

        # Give some admonition if the prefix is not set
        if prefix is None:
            self.log.warning("No prefix set, you have access to all parameters, be responsible :)")

        # Prefix all parameter names
        self.prefix = prefix + "/" if prefix else "/"


    def list(self) -> list:
        """List all parameters under the prefix in the AWS Parameter Store.

        Returns:
            list: A list of parameter names and details.
        """
        try:
            # Return the names of the parameters within the prefix
            if self.prefix == "/":
                response = self.ssm_client.describe_parameters()
            else:
                response = self.ssm_client.describe_parameters(
                    ParameterFilters=[{"Key": "Name", "Option": "BeginsWith", "Values": [self.prefix]}]
                )

            # Return the names of the parameters within the prefix
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

            # Remove any double // in the name
            name = name.replace("//", "/")

            # Retrieve the parameter from Parameter Store
            response = self.ssm_client.get_parameter(Name=name, WithDecryption=decrypt)
            value = response["Parameter"]["Value"]

            # Auto-detect and decompress if needed
            if value.startswith("COMPRESSED:"):
                # Base64 decode and decompress
                self.log.important(f"Decompressing parameter '{name}'...")
                compressed_value = base64.b64decode(value[len("COMPRESSED:") :])
                value = zlib.decompress(compressed_value).decode("utf-8")

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

            # Remove any double // in the name
            name = name.replace("//", "/")

            # Anything that's not a string gets converted to JSON
            if not isinstance(value, str):
                value = json.dumps(value)

            # Check size and compress if necessary
            if len(value) > 4096:
                self.log.warning(f"Parameter size exceeds 4KB: Compressing '{name}'...")
                compressed_value = zlib.compress(value.encode("utf-8"))
                encoded_value = "COMPRESSED:" + base64.b64encode(compressed_value).decode("utf-8")

                try:
                    # Add or update the compressed parameter in Parameter Store
                    self.ssm_client.put_parameter(Name=name, Value=encoded_value, Type="String", Overwrite=overwrite)
                    self.log.info(f"Parameter '{name}' added/updated successfully with compression.")
                    return
                except Exception as e:
                    self.log.critical(f"Failed to add/update compressed parameter '{name}': {e}")
                    raise

            # Add or update the parameter normally if under 4KB
            self.ssm_client.put_parameter(Name=name, Value=value, Type="String", Overwrite=overwrite)
            self.log.info(f"Parameter '{name}' added/updated successfully.")

        except Exception as e:
            self.log.critical(f"Failed to add/update parameter '{name}': {e}")
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
    # param_store.delete("test")
    # param_store.delete("my_data")

    # Now use a different prefix scope
    param_store = ParameterStore(prefix=None)
    print("Listing Parameters...")
    print(param_store.list())
