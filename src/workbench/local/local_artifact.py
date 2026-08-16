"""LocalArtifact: Base Class for all filesystem-backed Artifact classes.

Backs the Artifact metadata contract with an on-disk ``meta.json`` instead of
AWS tags, and carries no AWS session, bucket, or ARN.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Union

# Workbench Imports
from workbench.core.artifact import Artifact
from workbench.local import storage


class LocalArtifact(Artifact):
    """LocalArtifact: Base Class for all filesystem-backed Artifacts in Workbench"""

    # Subclasses set this to a key in storage.SUBDIRS
    artifact_type = None

    # Files hashed for content validation; subclasses extend as needed
    data_files = ("data.parquet",)

    def __init__(self, name: str, **kwargs):
        """Initialize the LocalArtifact Base Class

        Args:
            name (str): The Name of this artifact
        """
        super().__init__(name, **kwargs)
        self.path = storage.artifact_path(self.artifact_type, name)
        self.meta_path = os.path.join(self.path, "meta.json")
        self._meta_cache = None

    def exists(self) -> bool:
        """Does the Artifact exist on disk?"""
        return os.path.isfile(self.meta_path)

    def workbench_meta(self) -> Union[dict, None]:
        """Get the Workbench specific metadata for this Artifact

        Returns:
            Union[dict, None]: Dictionary of Workbench metadata for this Artifact
        """
        if self._meta_cache is None:
            try:
                with open(self.meta_path, "r") as fp:
                    self._meta_cache = json.load(fp)
            except (FileNotFoundError, json.JSONDecodeError):
                self._meta_cache = {}
        return self._meta_cache

    def upsert_workbench_meta(self, new_meta: dict):
        """Add Workbench specific metadata to this Artifact

        Args:
            new_meta (dict): Dictionary of NEW metadata to add
        """
        meta = {**self.workbench_meta(), **new_meta}
        os.makedirs(self.path, exist_ok=True)
        with open(self.meta_path, "w") as fp:
            json.dump(meta, fp, indent=4, default=str)
        self._meta_cache = meta

    def refresh_meta(self):
        """Refresh the Artifact's metadata"""
        self._meta_cache = None

    def delete_metadata(self, key_to_delete: str):
        """Delete specific metadata from this artifact

        Args:
            key_to_delete (str): Metadata key to delete
        """
        meta = dict(self.workbench_meta())
        if meta.pop(key_to_delete, None) is None:
            self.log.info(f"No Metadata found: {key_to_delete}...")
            return
        with open(self.meta_path, "w") as fp:
            json.dump(meta, fp, indent=4, default=str)
        self._meta_cache = meta

    def onboard(self) -> bool:
        """Onboard this Artifact into Workbench

        Returns:
            bool: True if the Artifact was successfully onboarded, False otherwise
        """
        self.upsert_workbench_meta({"workbench_status": "ready"})
        return True

    def details(self) -> dict:
        """Additional Details about this Artifact"""
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size(),
            "created": self.created(),
            "modified": self.modified(),
            **self.workbench_meta(),
        }

    def size(self) -> float:
        """Return the size of this artifact in MegaBytes"""
        return storage.dir_size_mb(self.path)

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        created = self.workbench_meta().get("workbench_created")
        if created:
            return datetime.fromisoformat(created)
        return self._mtime(self.meta_path)

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        newest = storage.newest_mtime(self.path)
        return datetime.fromtimestamp(newest, tz=timezone.utc) if newest else self.created()

    def hash(self) -> str:
        """Return the hash of this artifact, useful for content validation"""
        digest = hashlib.sha256()
        for file_name in self.data_files:
            file_path = os.path.join(self.path, file_name)
            if not os.path.isfile(file_path):
                continue
            with open(file_path, "rb") as fp:
                for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def delete(self):
        """Delete this artifact and everything under its directory"""
        if not os.path.isdir(self.path):
            self.log.warning(f"Local artifact {self.name} does not exist...")
            return
        self.log.important(f"Deleting local artifact {self.name} ({self.path})...")
        shutil.rmtree(self.path)
        self._meta_cache = None

    def _init_storage(self, input_name: str = "local"):
        """Internal: Create the artifact directory and stamp the initial metadata.

        Args:
            input_name (str): Name of this artifact's input (default: "local")
        """
        storage.local_root(create=True)
        os.makedirs(self.path, exist_ok=True)
        self.upsert_workbench_meta(
            {
                "workbench_created": datetime.now(timezone.utc).isoformat(),
                "workbench_input": input_name,
                "workbench_status": "ready",
                "workbench_tags": self.name,
            }
        )

    @staticmethod
    def _mtime(path: str) -> datetime:
        """Internal: Modification time of a single file as a UTC datetime.

        Args:
            path (str): File to stat

        Returns:
            datetime: UTC modification time (epoch if the file is missing)
        """
        try:
            return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        except OSError:
            return datetime.fromtimestamp(0, tz=timezone.utc)
