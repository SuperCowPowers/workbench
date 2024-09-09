"""ParameterStore: Manages SageWorks parameters in AWS Systems Manager Parameter Store."""

from typing import Union
import logging
import json
import zlib
import base64
from botocore.exceptions import ClientError

# SageWorks Imports
from sageworks.aws_service_broker.aws_session import AWSSession


class ParameterStore:
    """ParameterStore: Manages SageWorks parameters in AWS Systems Manager Parameter Store.

    Common Usage:
        ```
        params = ParameterStore()

        # List Parameters
        params.list()

        ['/sageworks/abalone_info',
         '/sageworks/my_data',
         '/sageworks/test',
         '/sageworks/pipelines/my_pipeline']

        # Add Key
        params.add("key", "value")
        value = params.get("key")

        # Add any data (lists, dictionaries, etc..)
        my_data = {"key": "value", "number": 4.2, "list": [1,2,3]}
        params.add("my_data", my_data)

        # Retrieve data
        return_value = params.get("my_data")
        pprint(return_value)

        {'key': 'value', 'list': [1, 2, 3], 'number': 4.2}

        # Delete parameters
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
        self.boto3_session = AWSSession().boto3_session

        # Create a Systems Manager (SSM) client for Parameter Store operations
        self.ssm_client = self.boto3_session.client("ssm")

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
            # Add the ParameterFilters if a prefix other than "/" is used
            params = {"MaxResults": 50}
            if self.prefix != "/":
                params["ParameterFilters"] = [{"Key": "Name", "Option": "BeginsWith", "Values": [self.prefix]}]

            # Initialize the list to collect parameter names
            all_parameters = []

            # Make the initial call to describe parameters
            response = self.ssm_client.describe_parameters(**params)

            # Aggregate the names from the initial response
            all_parameters.extend(param["Name"] for param in response["Parameters"])

            # Continue to paginate if there's a NextToken
            while "NextToken" in response:
                # Update the parameters with the NextToken for subsequent calls
                params["NextToken"] = response["NextToken"]
                response = self.ssm_client.describe_parameters(**params)

                # Aggregate the names from the subsequent responses
                all_parameters.extend(param["Name"] for param in response["Parameters"])

        except Exception as e:
            self.log.error(f"Failed to list parameters: {e}")

        # Return the aggregated list of parameter names
        return all_parameters

    def get(self, name: str, warn: bool = True, decrypt: bool = True) -> Union[str, list, dict, None]:
        """Retrieve a parameter value from the AWS Parameter Store.

        Args:
            name (str): The name of the parameter to retrieve.
            warn (bool): Whether to log a warning if the parameter is not found.
            decrypt (bool): Whether to decrypt secure string parameters.

        Returns:
            Union[str, list, dict, None]: The value of the parameter or None if not found.
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

        except ClientError as e:
            if e.response["Error"]["Code"] == "ParameterNotFound":
                if warn:
                    self.log.warning(f"Parameter '{name}' not found")
            else:
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
