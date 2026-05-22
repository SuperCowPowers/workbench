"""ParameterStoreCore: Endpoint-safe implementation for AWS Systems Manager Parameter Store.

This is the lightweight core class. The orchestration-side public API is
:class:`workbench.api.ParameterStore`, which subclasses this and pulls
the boto3 session from :class:`AWSAccountClamp` (refreshable credentials).
Endpoint code can instantiate ``ParameterStoreCore`` directly — it uses
:func:`workbench.core.cloud_platform.aws.boto_session.get_boto3_session`
which short-circuits to the container's ambient IAM role in service envs.
"""

from typing import Optional, Union
import logging
import json
import zlib
import time
import base64
from datetime import datetime
from botocore.exceptions import ClientError

# Workbench Imports
from workbench.core.cloud_platform.aws.boto_session import get_boto3_session
from workbench.utils.json_utils import CustomEncoder


class ParameterStoreCore:
    """ParameterStoreCore: Endpoint-safe core implementation for AWS Systems Manager Parameter Store.

    Note: Prefer the public :class:`workbench.api.ParameterStore` class for orchestration
    code (it uses refreshable credentials via AWSAccountClamp). This core class exists
    as a lower layer that endpoint code (and modules that would create a circular import
    through ``workbench.api``) can depend on.
    """

    def __init__(self, boto3_session=None):
        """ParameterStoreCore Init Method

        Args:
            boto3_session: (boto3.Session, optional): A boto3 session to use. Defaults to None.
        """
        self.log = logging.getLogger("workbench")

        # Initialize a Workbench Session (to assume the Workbench ExecutionRole)
        if boto3_session:
            self.boto3_session = boto3_session
        else:
            self.boto3_session = get_boto3_session()

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
            response = self._call_with_retry(self.ssm_client.describe_parameters, **params)

            # Aggregate the names from the initial response
            all_parameters.extend(param["Name"] for param in response["Parameters"])

            # Continue to paginate if there's a NextToken
            while "NextToken" in response:
                # Update the parameters with the NextToken for subsequent calls
                params["NextToken"] = response["NextToken"]
                response = self._call_with_retry(self.ssm_client.describe_parameters, **params)

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

    def _call_with_retry(self, func, **kwargs):
        """Call AWS API with exponential backoff on throttling."""
        max_retries = 5
        base_delay = 1
        for attempt in range(max_retries):
            try:
                return func(**kwargs)
            except ClientError as e:
                if e.response["Error"]["Code"] == "ThrottlingException" and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    self.log.warning(f"Throttled, retrying in {delay}s...")
                    time.sleep(delay)
                else:
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

    def last_modified(self, name: str) -> Optional[datetime]:
        """Return the LastModifiedDate of a parameter, or None if missing / unavailable.

        Useful for staleness checks against upstream resources that have their own
        modified-at timestamps (e.g. comparing a cached feature list's age to the
        endpoint it describes).

        Args:
            name: Parameter name (e.g. ``/workbench/feature_lists/smiles-to-2d-v1``).

        Returns:
            datetime (UTC, tz-aware) when the parameter was last written, or None
            if the parameter doesn't exist or the metadata call fails.
        """
        try:
            resp = self.ssm_client.describe_parameters(
                Filters=[{"Key": "Name", "Values": [name]}],
                MaxResults=1,
            )
            params = resp.get("Parameters", [])
            return params[0].get("LastModifiedDate") if params else None
        except Exception:
            # Staleness checks are an optimization — fail open and let the caller
            # fall back to trusting the cached value rather than hard-failing here.
            self.log.exception(f"Failed to read LastModifiedDate for parameter {name!r}")
            return None

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
        # Make sure prefix ends with a slash
        if not prefix.endswith("/"):
            prefix += "/"
        # List all parameters with the given prefix
        parameters = self.list(prefix=prefix)
        for param in parameters:
            self.delete(param)

    def __repr__(self):
        """Return a string representation of the ParameterStore object."""
        return "\n".join(self.list())
