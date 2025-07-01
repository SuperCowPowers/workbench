"""AWSParameterStore: Manages Workbench parameters in AWS Systems Manager Parameter Store."""

from typing import Union
import logging
import json
import zlib
import base64
from botocore.exceptions import ClientError

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_session import AWSSession
from workbench.utils.json_utils import CustomEncoder


class AWSParameterStore:
    """AWSParameterStore: Manages Workbench parameters in AWS Systems Manager Parameter Store.

    Common Usage:
        ```python
        params = AWSParameterStore()

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
        """AWSParameterStore Init Method"""
        self.log = logging.getLogger("workbench")

        # Initialize a Workbench Session (to assume the Workbench ExecutionRole)
        self.boto3_session = AWSSession().boto3_session

        # Create a Systems Manager (SSM) client for Parameter Store operations
        self.ssm_client = self.boto3_session.client("ssm")

    def list(self, prefix: str = None) -> list:
        """List all parameters in the AWS Parameter Store, optionally filtering by a prefix.

        Args:
            prefix (str, optional): A prefix to filter the parameters by. Defaults to None.

        Returns:
            list: A list of parameter names and details.
        """
        try:
            # Set up parameters for the query
            params = {"MaxResults": 50}

            # If a prefix is provided, add the 'ParameterFilters' for optimization
            if prefix:
                params["ParameterFilters"] = [{"Key": "Name", "Option": "BeginsWith", "Values": [prefix]}]

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
            return []

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
                # If parsing fails, return the value as is "hope for the best"
                return value

        except ClientError as e:
            if e.response["Error"]["Code"] == "ParameterNotFound":
                if warn:
                    self.log.warning(f"Parameter '{name}' not found")
            else:
                self.log.error(f"Failed to get parameter '{name}': {e}")
            return None

    def upsert(self, name: str, value, precision: int = 3):
        """Insert or update a parameter in the AWS Parameter Store.

        Args:
            name (str): The name of the parameter.
            value (str | list | dict): The value of the parameter.
            precision (int): The precision for float values in the JSON encoding.
        """
        try:
            # Convert to JSON and check if compression is needed
            json_value = json.dumps(value, cls=CustomEncoder, precision=precision)
            if len(json_value) <= 4096:
                # Store normally if under 4KB
                self._store_parameter(name, json_value)
                return

            # Need compression - log warning
            self.log.important(
                f"Parameter {name} exceeds 4KB ({len(json_value)} bytes): compressing and reducing precision..."
            )

            # Try compression with precision reduction
            compressed_value = self._compress_value(value)

            if len(compressed_value) <= 4096:
                self._store_parameter(name, compressed_value)
                return

            # Try clipping the data
            clipped_value = self._clip_data(value)
            compressed_clipped = self._compress_value(clipped_value)

            if len(compressed_clipped) <= 4096:
                self.log.warning(
                    f"Parameter {name} data clipped to 100 items/elements: ({len(compressed_clipped)} bytes)"
                )
                self._store_parameter(name, compressed_clipped)
                return

            # Still too large - give up
            self._handle_oversized_data(name, len(compressed_clipped))

        except Exception as e:
            self.log.critical(f"Failed to add/update parameter '{name}': {e}")
            raise

    @staticmethod
    def _compress_value(value) -> str:
        """Compress a value with precision reduction."""
        json_value = json.dumps(value, cls=CustomEncoder, precision=3)
        compressed = zlib.compress(json_value.encode("utf-8"), level=9)
        return "COMPRESSED:" + base64.b64encode(compressed).decode("utf-8")

    @staticmethod
    def _clip_data(value):
        """Clip data to reduce size, clip to first 100 items/elements."""
        if isinstance(value, dict):
            return dict(list(value.items())[:100])
        elif isinstance(value, list):
            return value[:100]
        return value

    def _store_parameter(self, name: str, value: str):
        """Store parameter in AWS Parameter Store."""
        self.ssm_client.put_parameter(Name=name, Value=value, Type="String", Overwrite=True)
        self.log.info(f"Parameter '{name}' added/updated successfully.")

    def _handle_oversized_data(self, name: str, size: int):
        """Handle data that's too large even after compression and clipping."""
        doc_link = "https://supercowpowers.github.io/workbench/api_classes/df_store"
        self.log.error(f"Compressed size {size} bytes, cannot store > 4KB")
        self.log.error(f"For larger data use the DFStore() class ({doc_link})")

    def delete(self, name: str):
        """Delete a parameter from the AWS Parameter Store.

        Args:
            name (str): The name of the parameter to delete.
        """
        try:
            # Delete the parameter from Parameter Store
            self.ssm_client.delete_parameter(Name=name)
            self.log.info(f"Parameter '{name}' deleted successfully.")
        except Exception as e:
            self.log.error(f"Failed to delete parameter '{name}': {e}")

    def delete_recursive(self, prefix: str):
        """Delete all parameters with a given prefix from the AWS Parameter Store.

        Args:
            prefix (str): The prefix of the parameters to delete.
        """
        # List all parameters with the given prefix
        parameters = self.list(prefix=prefix)
        for param in parameters:
            self.delete(param)

    def __repr__(self):
        """Return a string representation of the AWSParameterStore object."""
        return "\n".join(self.list())


if __name__ == "__main__":
    """Exercise the AWSParameterStore Class"""

    # Create a AWSParameterStore manager
    param_store = AWSParameterStore()

    # List the parameters
    print("Listing Parameters...")
    print(param_store.list())

    # Add a new parameter
    param_store.upsert("/workbench/test", "value")

    # Get the parameter
    print(f"Getting parameter 'test': {param_store.get('/workbench/test')}")

    # Add a dictionary as a parameter
    sample_dict = {"key": "str_value", "awesome_value": 4.2}
    param_store.upsert("/workbench/my_data", sample_dict)

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
