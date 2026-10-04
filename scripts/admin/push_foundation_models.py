"""Publish every registered foundation checkpoint to Workbench's public bucket (or an account mirror).

:func:`workbench.training.foundation_models.resolve_foundation_checkpoint` walks
local cache -> account mirror -> public bucket. This script fills the public bucket
(``workbench-public-data``, which needs SuperCowPowers write access), or with ``--bucket``
stages a mirror in an account that can't reach public S3::

    python scripts/admin/push_foundation_models.py                         # publish anything missing
    python scripts/admin/push_foundation_models.py --model monroe          # just one
    python scripts/admin/push_foundation_models.py --bucket my-wb-bucket   # an account mirror
    python scripts/admin/push_foundation_models.py --dry-run               # report, upload nothing

For each registered checkpoint it skips one already at its key with the registered size,
otherwise gets the file (the local cache, then the public bucket, then the entry's
``origin_url``), checks its md5 and size against the registry and that it loads as the
expected checkpoint, and uploads it plus a ``SOURCE.json`` provenance sidecar beside it.
Rerunning it is harmless. An object at the key with a different size is refused rather
than replaced (``--force`` replaces it). Torch is optional here: without it the structural
check is skipped with a warning (the md5/size check still runs).
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from workbench.training.foundation_models import (
    PUBLIC_BUCKET,
    file_hash,
    foundation_cache_dir,
    foundation_entry,
    known_foundation_models,
    resolve_foundation_checkpoint,
)

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def check_integrity(path: Path, entry: dict) -> dict:
    """Compare size and md5 against the registry's expected values.

    A truncated download, a captive-portal HTML page, or a checkpoint that changed
    under a stable record id all get caught here -- at the publishing gate, once,
    rather than inside a training job.

    Args:
        path (Path): Local checkpoint file.
        entry (dict): Registry entry for this foundation model.

    Returns:
        dict: {"md5", "size_bytes", "integrity"} for the sidecar.

    Raises:
        ValueError: If size or md5 disagrees with the registry.
    """
    size = path.stat().st_size
    md5 = file_hash(path, "md5")
    problems = []
    if size != entry["expected_size_bytes"]:
        problems.append(f"size {size} != expected {entry['expected_size_bytes']}")
    if md5 != entry["expected_md5"]:
        problems.append(f"md5 {md5} != expected {entry['expected_md5']}")
    if problems:
        raise ValueError(
            f"{path} failed its integrity check: "
            + "; ".join(problems)
            + f". If {entry['origin_url']} itself now serves different weights, add a new registry entry "
            "rather than publishing over this one."
        )
    print(f"  integrity OK: md5 {md5}, {size} bytes")
    return {"md5": md5, "size_bytes": size, "integrity": "verified"}


def verify_checkpoint(path: Path, entry: dict) -> dict:
    """Confirm the file loads as a torch checkpoint with the registry entry's keys.

    Args:
        path (Path): Local checkpoint file.
        entry (dict): Registry entry for this foundation model.

    Returns:
        dict: Details for the sidecar — the scalar settings under each non-weight key.
    """
    try:
        import torch
    except ImportError:
        log.warning("torch not installed — skipping the structural check (hash + upload only)")
        return {}

    try:
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as e:
        raise ValueError(
            f"{path} does not load as a torch checkpoint ({type(e).__name__}: {e}). "
            "A truncated download or an HTML error page will look like this."
        ) from None
    missing = [k for k in entry["checkpoint_keys"] if k not in ckpt]
    if missing:
        raise ValueError(f"{path} is not a {entry['filename']} checkpoint (missing {missing})")
    details = {
        key: {k: v for k, v in ckpt[key].items() if isinstance(v, (int, float, str, bool))}
        for key in entry["checkpoint_keys"]
        if key != "state_dict" and isinstance(ckpt[key], dict)
    }
    print(f"  checkpoint OK: {details}")
    return details


def target_size(client, bucket: str, key: str) -> int:
    """Byte size of s3://bucket/key, or None when it isn't there."""
    try:
        return client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def obtain(name: str, entry: dict) -> Path:
    """The checkpoint as a local file: the resolver's sources first, then the entry's origin URL.

    Args:
        name (str): Registered foundation model name.
        entry (dict): Its registry entry.

    Returns:
        Path: The local file (in the foundation cache).

    Raises:
        ValueError: If the origin download does not match the registry's md5.
    """
    try:
        return resolve_foundation_checkpoint(name)  # local cache -> account mirror -> public bucket
    except RuntimeError:
        pass

    dest = foundation_cache_dir() / entry["filename"]
    print(f"  downloading {entry['origin_url']} ...")
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(entry["origin_url"], tmp_path)
        if file_hash(tmp_path) != entry["expected_md5"]:
            raise ValueError(f"{entry['origin_url']} does not match the registry's md5 ({entry['expected_md5']})")
        shutil.move(str(tmp_path), dest)  # into the cache only once verified
    finally:
        tmp_path.unlink(missing_ok=True)
    return dest


def push(name: str, bucket: str, dry_run: bool, force: bool) -> None:
    """Publish one checkpoint plus its SOURCE.json sidecar, unless it is already there.

    Raises:
        ValueError: If the file fails its checks, or the target holds a different object.
    """
    entry = foundation_entry(name)
    key = entry["s3_key"]
    sidecar_key = f"{key.rsplit('/', 1)[0]}/SOURCE.json"
    client = boto3.client("s3")

    print(f"\n=== {name} ===")
    print(f"  target: s3://{bucket}/{key}")
    existing = target_size(client, bucket, key)
    if existing == entry["expected_size_bytes"] and not force:
        print("  already published, skipping")
        return
    if existing is not None and not force:
        raise ValueError(
            f"s3://{bucket}/{key} holds {existing} bytes, not {entry['expected_size_bytes']}: "
            "investigate, or rerun with --force to replace it"
        )

    local_file = obtain(name, entry)
    print(f"  local:  {local_file} ({local_file.stat().st_size / 1e6:.1f} MB)")
    integrity = check_integrity(local_file, entry)
    details = verify_checkpoint(local_file, entry)
    sidecar = {
        "model": name,
        "filename": entry["filename"],
        "origin_url": entry["origin_url"],
        "provenance_id": entry["provenance_id"],
        "description": entry["description"],
        "sha256": file_hash(local_file, "sha256"),
        **integrity,
        "uploaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "uploaded_by": "scripts/admin/push_foundation_models.py",
        **details,
    }

    if dry_run:
        print("  DRY RUN — nothing uploaded. Sidecar would be:")
        print(json.dumps(sidecar, indent=2))
        return

    client.upload_file(str(local_file), bucket, key)
    print(f"  uploaded {key}")
    client.put_object(
        Bucket=bucket, Key=sidecar_key, Body=json.dumps(sidecar, indent=2).encode(), ContentType="application/json"
    )
    print(f"  uploaded {sidecar_key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=known_foundation_models(), help="Only this checkpoint (default: all)")
    parser.add_argument(
        "--bucket", default=PUBLIC_BUCKET, help=f"An account bucket to stage a mirror in (default: {PUBLIC_BUCKET})"
    )
    parser.add_argument("--dry-run", action="store_true", help="Get and verify, but do not upload")
    parser.add_argument("--force", action="store_true", help="Upload even when the target already has the object")
    args = parser.parse_args()

    refused = []
    for name in [args.model] if args.model else known_foundation_models():
        try:
            push(name, args.bucket, args.dry_run, args.force)
        except ValueError as e:
            log.error(f"  REFUSING TO PUBLISH {name}: {e}")
            refused.append(name)
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
