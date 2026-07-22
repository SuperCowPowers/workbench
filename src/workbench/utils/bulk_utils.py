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
    # Item type matching is case-insensitive (e.g. "endpoint" and "Endpoint" both work).
    delete_methods = {
        "datasource": ("DataSource", DataSource.managed_delete),
        "featureset": ("FeatureSet", FeatureSet.managed_delete),
        "model": ("Model", Model.managed_delete),
        "endpoint": ("Endpoint", Endpoint.managed_delete),
    }
    for item_type, item_name in artifacts_to_delete:
        entry = delete_methods.get(item_type.lower())
        if entry is None:
            log.warning(f"Unknown item_type '{item_type}' for '{item_name}', skipping")
            continue
        label, managed_delete = entry
        log.info(f"Deleting {label}: {item_name}")
        managed_delete(item_name)


if __name__ == "__main__":
    # Test the bulk utils functions
    delete_list = [("DataSource", "abc"), ("FeatureSet", "abc_features")]
    bulk_delete(delete_list)
