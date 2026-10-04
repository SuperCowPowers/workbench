"""Foundation-model checkpoint registry and resolver (deliberately dep-free).

Pretrained checkpoints (CheMeleon, TabICL, Monroe) are published in Workbench's public
bucket, so every account resolves the same bytes with no staging. Resolution walks three
rungs:

1. **local cache** — ``~/.workbench/foundation/<filename>``; the only rung that survives
   *within* a container, and it is cold in every fresh training job.
2. **account mirror** — ``s3://$WORKBENCH_BUCKET/foundation-models/...``; optional, for an
   account that can't reach public S3 (a locked-down VPC).
3. **public bucket** — ``s3://workbench-public-data/foundation-models/...``; anonymous
   read, no AWS credentials needed.

A key carries its release (``foundation-models/<model>/<version>/<file>``) and is never
overwritten: new weights get a new key and a new registry entry in a Workbench release,
so a mirror holds the same bytes as the public copy. Every download is checked against
the entry's md5 before it enters the cache.

Publish to the public bucket, or stage a mirror, with
``scripts/admin/push_foundation_models.py``. A SageMaker training job has no site config,
so it gets ``WORKBENCH_BUCKET`` from the ``ModelTrainer`` environment set in
``features_to_model.py``.

No ``torch``/``chemprop``/``tabicl`` imports here on purpose: the training cores
(:mod:`workbench.training.chemprop_core`, :mod:`workbench.training.tabicl_core`) consume
this inside the training container, while the admin script imports the same registry
from a laptop that has none of the training deps installed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

log = logging.getLogger("workbench")

FOUNDATION_PREFIX = "foundation-models"

# Workbench's public bucket: anonymous read, the source of every registered checkpoint
PUBLIC_BUCKET = "workbench-public-data"
PUBLIC_BUCKET_REGION = "us-west-2"

# name -> checkpoint metadata. `s3_key` (and `filename`, the local cache name) carry the
# release, so new weights land beside the old copy instead of overwriting it. `origin_url`
# records where the file was published, for the staging sidecar.
FOUNDATION_MODELS = {
    "chemeleon": {
        "filename": "chemeleon_mp-15460715.pt",
        "s3_key": f"{FOUNDATION_PREFIX}/chemeleon/15460715/chemeleon_mp.pt",
        "origin_url": "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
        "provenance_id": "zenodo-15460715",
        "description": "CheMeleon MPNN foundation weights (Zenodo record 15460715)",
        "expected_md5": "6a80b54fdb7de37ef0374d302f01e8ce",
        "expected_size_bytes": 34859448,
        # Top-level keys a valid checkpoint carries, checked at staging time
        "checkpoint_keys": ["hyper_parameters", "state_dict"],
    },
    "tabicl": {
        "filename": "tabicl-regressor-v2-20260212.ckpt",
        "s3_key": f"{FOUNDATION_PREFIX}/tabicl/4dcd344e/tabicl-regressor-v2-20260212.ckpt",
        "origin_url": (
            "https://huggingface.co/jingang/TabICL/resolve/"
            "4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-regressor-v2-20260212.ckpt"
        ),
        "provenance_id": "hf-jingang-TabICL-4dcd344e",
        "description": "TabICL v2 regressor checkpoint (Hugging Face jingang/TabICL, revision 4dcd344e)",
        "expected_md5": "e9b7c522e50a3fc6ad5cf3486dcebc46",
        "expected_size_bytes": 114324594,
        "checkpoint_keys": ["config", "state_dict"],
    },
    "monroe": {
        "filename": "monroe-57238ed-weights.pt",
        "s3_key": f"{FOUNDATION_PREFIX}/monroe/57238ed/weights.pt",
        "origin_url": (
            "https://github.com/blazejba/monroe/raw/57238edfffea03808abe761a00cd9a75fa41bb95/checkpoint/weights.pt"
        ),
        "provenance_id": "github-blazejba-monroe-57238ed",
        "description": "Monroe molecular encoder weights (GitHub blazejba/monroe, commit 57238ed)",
        "expected_md5": "2d4509e483aabf46781709146084b47e",
        "expected_size_bytes": 300057823,
        # A flat state dict (CUDA tensors): these are tensor names, not sections
        "checkpoint_keys": ["encoder.missing_fill", "encoder.pooling.value_proj.weight"],
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
    # to read s3://change_me/... just wastes a round trip before the public bucket.
    placeholders = {"change_me", "env-will-overwrite", ""}

    bucket = os.environ.get("WORKBENCH_BUCKET")
    if bucket and bucket not in placeholders:
        return bucket
    try:
        from workbench.utils.config_manager import ConfigManager

        bucket = ConfigManager().get_config("WORKBENCH_BUCKET")
    except Exception as e:  # no config file, unreadable, etc. — the mirror rung just gets skipped
        log.info(f"No WORKBENCH_BUCKET from config ({e}); skipping the account mirror")
        return None
    if bucket in placeholders:
        log.info(f"WORKBENCH_BUCKET is unset/placeholder ({bucket!r}); skipping the account mirror")
        return None
    return bucket


def foundation_cache_dir() -> Path:
    """Local checkpoint cache directory (created if absent)."""
    cache_dir = Path.home() / ".workbench" / "foundation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def file_hash(path: Path, algorithm: str = "md5") -> str:
    """Streaming hash of a file (checkpoints are hundreds of MB).

    Args:
        path (Path): File to hash.
        algorithm (str, optional): Any :mod:`hashlib` name. Defaults to "md5".

    Returns:
        str: Hex digest.
    """
    digest = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_s3(bucket: str, key: str, dest: Path, expected_md5: str = None, unsigned: bool = False) -> bool:
    """Download s3://bucket/key to dest atomically, checking its md5 when one is expected.

    Args:
        bucket (str): Bucket to read.
        key (str): Object key.
        dest (Path): Local destination; written only once the download is complete and verified.
        expected_md5 (str, optional): md5 the object must match.
        unsigned (bool, optional): Anonymous request (the public bucket). Defaults to False.

    Returns:
        bool: True when dest holds the verified object.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError

    client = (
        boto3.client("s3", region_name=PUBLIC_BUCKET_REGION, config=Config(signature_version=UNSIGNED))
        if unsigned
        else boto3.client("s3")
    )
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        client.download_file(bucket, key, str(tmp_path))
        if expected_md5 and file_hash(tmp_path) != expected_md5:
            log.warning(f"s3://{bucket}/{key} does not match the registry's md5 ({expected_md5}); not using it")
            return False
        shutil.move(str(tmp_path), dest)  # atomic within the same filesystem
        return True
    except ClientError as e:
        missing = e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey")
        (log.info if missing else log.warning)(f"Foundation checkpoint not available at s3://{bucket}/{key}: {e}")
        return False
    except Exception as e:
        log.warning(f"Foundation checkpoint not available at s3://{bucket}/{key}: {e}")
        return False
    finally:
        tmp_path.unlink(missing_ok=True)  # no half-downloads left in the cache dir


def fetch_s3_checkpoint(s3_uri: str) -> Path:
    """Download an explicit ``s3://`` checkpoint into the local cache.

    No registry entry and no other rungs: an explicit URI is the caller saying
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


def resolve_foundation_checkpoint(name: str) -> Path:
    """Local path to a foundation checkpoint, fetching it if needed.

    Walks local cache -> account mirror -> public bucket (see the module docstring).

    Args:
        name (str): Foundation model name, e.g. "CheMeleon".

    Returns:
        pathlib.Path: Path to the local checkpoint file.

    Raises:
        RuntimeError: If neither the mirror nor the public bucket has it.
    """
    entry = foundation_entry(name)
    key, expected_md5 = entry["s3_key"], entry.get("expected_md5")
    local_path = foundation_cache_dir() / entry["filename"]

    if local_path.exists():
        print(f"  Using cached checkpoint: {local_path}")
        return local_path

    # The account mirror first (an account that stages one may not reach public S3), then the public copy
    mirror = workbench_bucket()
    sources = ([(mirror, False)] if mirror else []) + [(PUBLIC_BUCKET, True)]
    for bucket, unsigned in sources:
        print(f"  Fetching checkpoint from s3://{bucket}/{key} ...")
        if _download_s3(bucket, key, local_path, expected_md5=expected_md5, unsigned=unsigned):
            print(f"  Downloaded to {local_path}")
            return local_path

    raise RuntimeError(f"Could not obtain foundation checkpoint '{name}' from s3://{PUBLIC_BUCKET}/{key}")
