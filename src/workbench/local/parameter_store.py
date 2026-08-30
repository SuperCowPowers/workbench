"""ParameterStore: Filesystem-backed key/value store mirroring the AWS Parameter Store.

Parameters are JSON files under ``<WORKBENCH_LOCAL_PATH>/parameter_store``, with the
parameter path becoming the directory path. Values round-trip through the same JSON
encoder the AWS store uses, so a dict written locally reads back the same shape after
publishing.

Local files have no 4KB ceiling and nothing to decrypt, so the compression and
decryption the AWS store needs have no counterpart here.
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Optional, Union

from workbench.local.storage import local_root
from workbench.utils.json_utils import CustomEncoder

# Parameters live under their own subdirectory of the local root.
SUBDIR = "parameter_store"


class ParameterStore:
    """ParameterStore: Manages Workbench parameters on the local filesystem.

    Common Usage:
        ```python
        params = ParameterStore()

        # List Parameters
        params.list()

        # Add Key
        params.upsert("key", "value")
        value = params.get("key")

        # Add any data (lists, dictionaries, etc..)
        params.upsert("my_data", {"key": "value", "number": 4.2, "list": [1, 2, 3]})

        # Delete parameters
        params.delete("my_data")
        ```
    """

    def __init__(self):
        """ParameterStore Init Method"""
        self.log = logging.getLogger("workbench")
        self.root = os.path.join(local_root(), SUBDIR)

    @staticmethod
    def _normalize(name: str) -> str:
        """Put a parameter name into canonical form: absolute, single-slashed, no trailing slash.

        Matches the AWS store so the same parameter name addresses the same thing in
        either one.
        """
        return re.sub(r"/+", "/", f"/{name}").rstrip("/")

    def _path(self, name: str) -> str:
        """Filesystem path backing a parameter name.

        Raises:
            ValueError: If the name would escape the store's root.
        """
        relative = self._normalize(name).lstrip("/")
        if not relative or any(part in ("..", ".") for part in relative.split("/")):
            raise ValueError(f"Invalid parameter name: {name!r}")
        return os.path.join(self.root, f"{relative}.json")

    def _name(self, path: str) -> str:
        """Parameter name for a backing file path (the inverse of ``_path``)."""
        relative = os.path.relpath(path, self.root)
        return self._normalize(os.path.splitext(relative)[0])

    def list(self, prefix: str = None, details: bool = False) -> list:
        """List all parameters in the store, optionally filtering by a prefix.

        Args:
            prefix (str, optional): A hierarchy path to list under, e.g. "/workbench/models".
                Matches whole path segments, not arbitrary string prefixes. The leading
                slash is optional. Defaults to None.
            details (bool, optional): Return ``{"name", "modified"}`` dicts instead of bare
                names. Defaults to False.

        Returns:
            list: Parameter names, or dicts of name + last-modified when details is True.
        """
        if not os.path.isdir(self.root):
            return []
        prefix = self._normalize(prefix) if prefix else None
        entries = []
        for dirpath, _, filenames in os.walk(self.root):
            for filename in filenames:
                if not filename.endswith(".json"):
                    continue
                path = os.path.join(dirpath, filename)
                name = self._name(path)
                # Whole segments only, so "/a/bc" doesn't match a "/a/b" prefix
                if prefix and not (name == prefix or name.startswith(f"{prefix}/")):
                    continue
                entries.append({"name": name, "modified": self._mtime(path)} if details else name)
        return sorted(entries, key=lambda e: e["name"] if details else e)

    @staticmethod
    def _mtime(path: str) -> datetime:
        """Modification time of a backing file, as a tz-aware UTC datetime."""
        return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)

    def get(self, name: str, warn: bool = True) -> Union[str, list, dict, None]:
        """Retrieve a parameter value from the store.

        Args:
            name (str): The name of the parameter to retrieve (leading slash optional).
            warn (bool): Whether to log a warning if the parameter is not found.

        Returns:
            The parameter value, or None if it doesn't exist.
        """
        path = self._path(name)
        if not os.path.isfile(path):
            if warn:
                self.log.warning(f"Parameter '{self._normalize(name)}' not found")
            return None
        try:
            with open(path, "r") as fp:
                value = fp.read()
        except OSError as e:
            self.log.error(f"Failed to get parameter '{name}': {e}")
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Same fallback as the AWS store: hand back whatever is there
            return value

    def upsert(self, name: str, value, precision: int = 3):
        """Insert or update a parameter in the store.

        Args:
            name (str): The name of the parameter (leading slash optional).
            value (str | list | dict): The value of the parameter.
            precision (int): The precision for float values in the JSON encoding.
        """
        path = self._path(name)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fp:
                fp.write(json.dumps(value, cls=CustomEncoder, precision=precision))
        except (OSError, TypeError) as e:
            self.log.critical(f"Failed to add/update parameter '{name}': {e}")
            raise

    def last_modified(self, name: str) -> Optional[datetime]:
        """Return when a parameter was last written, or None if it doesn't exist.

        Args:
            name (str): Parameter name (leading slash optional).

        Returns:
            datetime (UTC, tz-aware) when the parameter was last written, or None.
        """
        path = self._path(name)
        return self._mtime(path) if os.path.isfile(path) else None

    def delete(self, name: str):
        """Delete a parameter from the store.

        Args:
            name (str): The name of the parameter to delete (leading slash optional).
        """
        path = self._path(name)
        try:
            os.remove(path)
            self.log.info(f"Parameter '{self._normalize(name)}' deleted successfully.")
        except FileNotFoundError:
            self.log.error(f"Failed to delete parameter '{self._normalize(name)}': not found")
        except OSError as e:
            self.log.error(f"Failed to delete parameter '{name}': {e}")

    def delete_recursive(self, prefix: str):
        """Delete every parameter under a given path.

        Deletes the parameters *under* the path. A parameter whose name is exactly
        ``prefix`` is a sibling of that path, not a child, so delete it with
        :meth:`delete`.

        Args:
            prefix (str): Path to delete under (leading slash optional).
        """
        for name in self.list(prefix=prefix):
            if name != self._normalize(prefix):
                self.delete(name)
        # Prune the directory the parameters lived in, if it's now empty
        directory = os.path.join(self.root, self._normalize(prefix).lstrip("/"))
        if os.path.isdir(directory) and not os.listdir(directory):
            shutil.rmtree(directory, ignore_errors=True)

    def __repr__(self):
        """Return a string representation of the ParameterStore object."""
        return "\n".join(self.list())
