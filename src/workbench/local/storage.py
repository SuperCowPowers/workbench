"""Storage layout for local artifacts.

Root comes from the ``WORKBENCH_LOCAL_PATH`` config key (default
``~/.workbench/local``). Each artifact type owns a subdirectory, and each
artifact owns a directory under that holding its data plus a ``meta.json``.
"""

import os
from typing import Optional

from workbench.utils.config_manager import ConfigManager

# Artifact type -> subdirectory under the local root
SUBDIRS = {
    "data_source": "data_sources",
    "feature_set": "feature_sets",
    "model": "models",
    "endpoint": "endpoints",
}


def local_root(create: bool = False) -> str:
    """Get the storage root for local artifacts.

    Args:
        create (bool): Create the root and its subdirectories if missing (default: False)

    Returns:
        str: Path to the local artifact storage root
    """
    root = ConfigManager().get_config("WORKBENCH_LOCAL_PATH")
    if create:
        for subdir in SUBDIRS.values():
            os.makedirs(os.path.join(root, subdir), exist_ok=True)
    return root


def root_exists() -> bool:
    """Does the local storage root already exist?

    Callers that want a human to approve the directory (the Bosco agent) check
    this before touching anything; the artifact classes themselves are
    non-interactive and create the root on demand.

    Returns:
        bool: True if the local storage root exists
    """
    return os.path.isdir(local_root())


def artifact_path(artifact_type: str, name: str, create: bool = False) -> str:
    """Get the directory for a single local artifact.

    Args:
        artifact_type (str): One of the keys in SUBDIRS
        name (str): The artifact name
        create (bool): Create the directory if missing (default: False)

    Returns:
        str: Path to the artifact's directory
    """
    path = os.path.join(local_root(), SUBDIRS[artifact_type], name)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def list_artifacts(artifact_type: str) -> list[str]:
    """List the names of local artifacts of the given type.

    Args:
        artifact_type (str): One of the keys in SUBDIRS

    Returns:
        list[str]: Sorted artifact names
    """
    type_dir = os.path.join(local_root(), SUBDIRS[artifact_type])
    if not os.path.isdir(type_dir):
        return []
    return sorted(d.name for d in os.scandir(type_dir) if d.is_dir())


def dir_size_mb(path: str) -> float:
    """Total size of a directory tree in MegaBytes.

    Args:
        path (str): Directory to measure

    Returns:
        float: Size in MB (0.0 if the directory is missing)
    """
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def newest_mtime(path: str) -> Optional[float]:
    """Most recent modification time in a directory tree.

    Args:
        path (str): Directory to scan

    Returns:
        Optional[float]: Unix timestamp, or None if the directory is missing/empty
    """
    times = [os.path.getmtime(os.path.join(dirpath, f)) for dirpath, _, filenames in os.walk(path) for f in filenames]
    return max(times) if times else None
