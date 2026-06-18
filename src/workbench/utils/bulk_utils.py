"""Bulk Operation Utilities"""

import logging

# Workbench imports
from workbench.api import DataSource, FeatureSet, Model, Endpoint

log = logging.getLogger("workbench")


def bulk_delete(artifacts_to_delete: list[tuple[str, str]]):
    """Bulk delete various artifacts

    Bypasses artifact object creation and calls each class's managed_delete()
    directly. This is equivalent to instantiating the artifact and calling
    .delete() (e.g. Model(name).delete()), with one caveat: DataSource deletion
    targets the default "workbench" database, so a DataSource in a non-default
    database must be deleted via its instance.

    Args:
        artifacts_to_delete (list[tuple[str, str]]): A list of tuples of the form (item_type, item_name)
    """
    for item_type, item_name in artifacts_to_delete:
        if item_type == "DataSource":
            log.info(f"Deleting DataSource: {item_name}")
            DataSource.managed_delete(item_name)
        elif item_type == "FeatureSet":
            log.info(f"Deleting FeatureSet: {item_name}")
            FeatureSet.managed_delete(item_name)
        elif item_type == "Model":
            log.info(f"Deleting Model: {item_name}")
            Model.managed_delete(item_name)
        elif item_type == "Endpoint":
            log.info(f"Deleting Endpoint: {item_name}")
            Endpoint.managed_delete(item_name)


if __name__ == "__main__":
    # Test the bulk utils functions
    delete_list = [("DataSource", "abc"), ("FeatureSet", "abc_features")]
    bulk_delete(delete_list)
