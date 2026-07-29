"""Foundation-model checkpoint registry and resolver (deliberately dep-free).

Warm-start weights (CheMeleon and friends) are published on the public internet,
but a training job must never *depend* on that, so checkpoints are staged in the
Workbench bucket and resolution walks three rungs:

1. **local cache** — ``~/.chemprop/foundation/<filename>``; the only rung that
   survives *within* a container, and it is cold in every fresh training job.
2. **Workbench S3** — ``s3://$WORKBENCH_BUCKET/foundation-models/...``; the
   durable copy, inside the account, no public dependency.
3. **origin URL** — the public internet, last resort, warns loudly. Pass
   ``allow_origin=False`` to make a missing S3 copy a hard error instead.

Populate rung 2 with ``scripts/admin/push_chemeleon_models.py``. A SageMaker training
job has no site config, so it gets ``WORKBENCH_BUCKET`` from the ``ModelTrainer``
environment set in ``features_to_model.py``.

No ``torch``/``chemprop`` imports here on purpose: :mod:`workbench.training.chemprop_core`
consumes this inside the training container, while the admin script imports the
same registry from a laptop that has none of the training deps installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

log = logging.getLogger("workbench")

FOUNDATION_PREFIX = "foundation-models"

# name -> checkpoint metadata. `s3_key` pins the provenance id (Zenodo record) in
# the path so a re-release lands beside the old copy instead of overwriting it.
FOUNDATION_MODELS = {
    "chemeleon": {
        "filename": "chemeleon_mp.pt",
        "s3_key": f"{FOUNDATION_PREFIX}/chemeleon/15460715/chemeleon_mp.pt",
        "origin_url": "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
        "provenance_id": "zenodo-15460715",
        "description": "CheMeleon MPNN foundation weights (Zenodo record 15460715)",
        # Expected integrity of the origin file, checked at staging time only (see
        # scripts/admin/push_chemeleon_models.py). Reported by an operator, not
        # verified in-account -- treat a mismatch as "investigate", not "impossible".
        "expected_md5": "6a80b54fdb7de37ef0374d302f01e8ce",
        "expected_size_bytes": 34859448,
    },
}


def known_foundation_models() -> list:
    """Names accepted by :func:`resolve_foundation_checkpoint`."""
    return sorted(FOUNDATION_MODELS)


def foundation_entry(name: str) -> dict:
    """Registry entry for ``name`` (case-insensitive).

    Args:
        name (str): Foundation model name, e.g. "CheMeleon".

    Returns:
        dict: The registry entry.

    Raises:
        ValueError: If the name is not registered.
    """
    entry = FOUNDATION_MODELS.get(name.lower())
    if entry is None:
        raise ValueError(f"Unknown foundation model: {name}. Known: {known_foundation_models()}")
    return entry


def workbench_bucket() -> str:
    """The Workbench bucket, from the ENV var first, then the config file.

    Returns:
        str: Bucket name, or None if neither source has it (a bare training
            container with no Workbench config).
    """
    # Placeholders from the bootstrap config (config_manager._load_bootstrap_config) are
    # NOT a bucket -- a container with no Workbench config yields "change_me", and trying
    # to read s3://change_me/... just wastes a round trip before the origin fallback.
    placeholders = {"change_me", "env-will-overwrite", ""}

    bucket = os.environ.get("WORKBENCH_BUCKET")
    if bucket and bucket not in placeholders:
        return bucket
    try:
        from workbench.utils.config_manager import ConfigManager

        bucket = ConfigManager().get_config("WORKBENCH_BUCKET")
    except Exception as e:  # no config file, unreadable, etc. — S3 rung just gets skipped
        log.warning(f"Could not resolve WORKBENCH_BUCKET from config: {e}")
        return None
    if bucket in placeholders:
        log.warning(f"WORKBENCH_BUCKET is unset/placeholder ({bucket!r}) — skipping the staged S3 copy")
        return None
    return bucket


def foundation_s3_uri(name: str, bucket: str = None) -> str:
    """Full ``s3://`` URI of the staged checkpoint.

    Args:
        name (str): Foundation model name.
        bucket (str, optional): Override the Workbench bucket.

    Returns:
        str: The S3 URI, or None if no bucket could be resolved.
    """
    bucket = bucket or workbench_bucket()
    if not bucket:
        return None
    return f"s3://{bucket}/{foundation_entry(name)['s3_key']}"


def foundation_cache_dir() -> Path:
    """Local checkpoint cache directory (created if absent)."""
    cache_dir = Path.home() / ".chemprop" / "foundation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_s3(bucket: str, key: str, dest: Path) -> bool:
    """Download s3://bucket/key to dest atomically. True on success."""
    import boto3

    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        boto3.client("s3").download_file(bucket, key, str(tmp_path))
        shutil.move(str(tmp_path), dest)  # atomic within the same filesystem
        return True
    except Exception as e:
        log.warning(f"Foundation checkpoint not available at s3://{bucket}/{key}: {e}")
        return False
    finally:
        tmp_path.unlink(missing_ok=True)  # no half-downloads left in the cache dir


def _download_origin(url: str, dest: Path) -> bool:
    """Download the public origin URL to dest atomically. True on success."""
    import urllib.request

    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        shutil.move(str(tmp_path), dest)
        return True
    except Exception as e:
        log.warning(f"Origin download failed for {url}: {e}")
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def fetch_s3_checkpoint(s3_uri: str) -> Path:
    """Download an explicit ``s3://`` checkpoint into the local cache.

    No origin rung and no registry entry: an explicit URI is the caller saying
    exactly which artifact they want, so a miss is a hard error. The cache
    filename is prefixed with a hash of the URI, so two different staged
    checkpoints that share a basename cannot shadow each other.

    Args:
        s3_uri (str): Full ``s3://bucket/key`` of the checkpoint.

    Returns:
        pathlib.Path: Path to the local file.

    Raises:
        ValueError: If the URI is malformed.
        RuntimeError: If the object could not be downloaded.
    """
    import hashlib

    bucket, _, key = s3_uri[len("s3://") :].partition("/")
    if not bucket or not key:
        raise ValueError(f"Malformed S3 URI: {s3_uri}")

    tag = hashlib.md5(s3_uri.encode()).hexdigest()[:8]
    local_path = foundation_cache_dir() / f"{tag}_{key.rsplit('/', 1)[-1]}"
    if local_path.exists():
        print(f"  Using cached checkpoint: {local_path}")
        return local_path

    print(f"  Fetching checkpoint from {s3_uri} ...")
    if not _download_s3(bucket, key, local_path):
        raise RuntimeError(f"Could not download foundation checkpoint from {s3_uri}")
    print(f"  Downloaded to {local_path}")
    return local_path


def resolve_foundation_checkpoint(name: str, allow_origin: bool = True) -> Path:
    """Local path to a foundation checkpoint, fetching it if needed.

    Walks local cache -> Workbench S3 -> origin URL (see the module docstring).

    Args:
        name (str): Foundation model name, e.g. "CheMeleon".
        allow_origin (bool, optional): Allow the public-internet fallback.
            Defaults to True. Set False to require the staged S3 copy.

    Returns:
        pathlib.Path: Path to the local checkpoint file.

    Raises:
        RuntimeError: If every rung fails.
    """
    entry = foundation_entry(name)
    local_path = foundation_cache_dir() / entry["filename"]

    if local_path.exists():
        print(f"  Using cached checkpoint: {local_path}")
        return local_path

    bucket = workbench_bucket()
    if bucket:
        key = entry["s3_key"]
        print(f"  Fetching checkpoint from s3://{bucket}/{key} ...")
        if _download_s3(bucket, key, local_path):
            print(f"  Downloaded to {local_path}")
            return local_path
    else:
        log.warning("No WORKBENCH_BUCKET resolved — skipping the staged S3 copy")

    if not allow_origin:
        raise RuntimeError(
            f"Foundation checkpoint '{name}' not staged in S3 ({foundation_s3_uri(name)}) and "
            f"allow_origin=False. Stage it with scripts/admin/push_chemeleon_models.py"
        )

    url = entry["origin_url"]
    log.warning(f"Falling back to public origin for '{name}': {url} — stage it in S3 to avoid this")
    if _download_origin(url, local_path):
        print(f"  Downloaded to {local_path}")
        return local_path

    raise RuntimeError(
        f"Could not obtain foundation checkpoint '{name}' from cache, " f"{foundation_s3_uri(name)}, or {url}"
    )
