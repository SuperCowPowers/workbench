"""Bulk Operation Utilities"""

import logging

# SageWorks imports
from sageworks.api import DataSource, FeatureSet, Model, Endpoint

log = logging.getLogger("sageworks")


def bulk_delete(artifacts_to_delete: list[tuple[str, str]]):
    """Bulk delete various artifacts

    Args:
        artifacts_to_delete (list[tuple[str, str]]): A list of tuples of the form (item_type, item_uuid)
    """
    for item_type, item_uuid in artifacts_to_delete:
        if item_type == "DataSource":
            log.info(f"Deleting DataSource: {item_uuid}")
            DataSource.managed_delete(item_uuid)
        elif item_type == "FeatureSet":
            log.info(f"Deleting FeatureSet: {item_uuid}")
            FeatureSet.managed_delete(item_uuid)
        elif item_type == "Model":
            log.info(f"Deleting Model: {item_uuid}")
            Model.managed_delete(item_uuid)
        elif item_type == "Endpoint":
            log.info(f"Deleting Endpoint: {item_uuid}")
            Endpoint.managed_delete(item_uuid)


if __name__ == "__main__":
    # Test the bulk utils functions
    delete_list = [("DataSource", "abc"), ("FeatureSet", "abc_features")]
    bulk_delete(delete_list)
