"""ParameterStoreCore: Manages Workbench parameters in a Cloud Based Parameter Store."""

import logging
from datetime import datetime
from typing import Optional

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

# Workbench Bridges Import
from workbench_bridges.api import ParameterStore as BridgesParameterStore


class ParameterStoreCore(BridgesParameterStore):
    """ParameterStoreCore: Manages Workbench parameters in a Cloud Based Parameter Store.

    Common Usage:
        ```python
        params = ParameterStoreCore()

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
        """ParameterStoreCore Init Method"""
        session = AWSAccountClamp().boto3_session

        # Initialize parent with workbench config
        super().__init__(boto3_session=session)
        self.log = logging.getLogger("workbench")

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


if __name__ == "__main__":
    """Exercise the ParameterStoreCore Class"""

    # Create a ParameterStoreCore manager
    param_store = ParameterStoreCore()

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

    # Recursive delete test
    param_store.upsert("/workbench/test/test1", "value1")
    param_store.upsert("/workbench/test/test2", "value2")
    param_store.delete_recursive("workbench/test/")
