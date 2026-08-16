"""LocalMeta: Listings for the artifacts in local storage.

A directory glob plus a ``meta.json`` read per artifact. No caching tier and no
artifact objects constructed, so listing stays cheap.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

# Workbench Imports
from workbench.core.artifact import Artifact
from workbench.local import storage


class LocalMeta:
    """LocalMeta: Workbench Local Metadata Class

    Common Usage:
        ```python
        meta = LocalMeta()
        meta.data_sources()
        meta.feature_sets()
        meta.models()
        meta.endpoints()
        ```
    """

    def __init__(self):
        """Initialize the LocalMeta class"""
        self.log = Artifact.log

    def data_sources(self) -> pd.DataFrame:
        """Get a summary of the local Data Sources

        Returns:
            pd.DataFrame: A summary of the local Data Sources
        """
        return self._summary("data_source", extra={"Rows": "num_rows", "Columns": "num_columns"})

    def feature_sets(self) -> pd.DataFrame:
        """Get a summary of the local Feature Sets

        Returns:
            pd.DataFrame: A summary of the local Feature Sets
        """
        return self._summary("feature_set", extra={"Rows": "num_rows", "Columns": "num_columns", "Id": "id_column"})

    def models(self) -> pd.DataFrame:
        """Get a summary of the local Models

        Returns:
            pd.DataFrame: A summary of the local Models
        """
        return self._summary("model", extra={"Type": "model_type", "Framework": "model_framework"})

    def endpoints(self) -> pd.DataFrame:
        """Get a summary of the local Endpoints

        Returns:
            pd.DataFrame: A summary of the local Endpoints
        """
        return self._summary("endpoint", extra={"Model": "model_name"})

    def _summary(self, artifact_type: str, extra: dict = None) -> pd.DataFrame:
        """Internal: Build a summary DataFrame for one artifact type.

        Args:
            artifact_type (str): One of the keys in storage.SUBDIRS
            extra (dict, optional): Column name -> metadata key, appended per type

        Returns:
            pd.DataFrame: One row per artifact
        """
        extra = extra or {}
        columns = ["Name", "Health", "Owner", "Created", "Modified", "Status", "Input", "Tags"] + list(extra)

        rows = []
        for name in storage.list_artifacts(artifact_type):
            path = storage.artifact_path(artifact_type, name)
            meta = self._read_meta(path)
            if meta is None:
                continue

            row = {
                "Name": name,
                "Health": ", ".join(self._split(meta.get("workbench_health_tags"))) or "healthy",
                "Owner": meta.get("workbench_owner", "unknown"),
                "Created": self._created(meta, path),
                "Modified": self._modified(path),
                "Status": meta.get("workbench_status", "unknown"),
                "Input": meta.get("workbench_input", "unknown"),
                "Tags": ", ".join(self._split(meta.get("workbench_tags"))),
            }
            # num_columns isn't stored; it's the length of the stored column list
            for column, key in extra.items():
                row[column] = len(meta.get("columns", [])) if key == "num_columns" else meta.get(key)
            rows.append(row)

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _read_meta(path: str) -> dict:
        """Internal: Read an artifact's meta.json

        Args:
            path (str): The artifact directory

        Returns:
            dict: The metadata, or None if unreadable
        """
        try:
            with open(os.path.join(path, "meta.json"), "r") as fp:
                return json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    @staticmethod
    def _split(tags: str) -> list[str]:
        """Internal: Split a delimited tag string

        Args:
            tags (str): The delimited tag string

        Returns:
            list[str]: The individual tags
        """
        return tags.split(Artifact.tag_delimiter) if tags else []

    @staticmethod
    def _created(meta: dict, path: str) -> datetime:
        """Internal: Creation time from metadata, falling back to the meta.json mtime

        Args:
            meta (dict): The artifact metadata
            path (str): The artifact directory

        Returns:
            datetime: The creation time
        """
        created = meta.get("workbench_created")
        if created:
            return datetime.fromisoformat(created)
        return datetime.fromtimestamp(os.path.getmtime(os.path.join(path, "meta.json")), tz=timezone.utc)

    @staticmethod
    def _modified(path: str) -> datetime:
        """Internal: Most recent modification time in the artifact directory

        Args:
            path (str): The artifact directory

        Returns:
            datetime: The modification time
        """
        newest = storage.newest_mtime(path)
        return datetime.fromtimestamp(newest, tz=timezone.utc) if newest else None


if __name__ == "__main__":
    """Exercise the LocalMeta Class"""
    meta = LocalMeta()
    print("Data Sources:")
    print(meta.data_sources())
    print("\nFeature Sets:")
    print(meta.feature_sets())
    print("\nModels:")
    print(meta.models())
    print("\nEndpoints:")
    print(meta.endpoints())
