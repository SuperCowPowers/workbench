"""Stage foundation-model checkpoints in the Workbench bucket.

:func:`workbench.training.chemprop_core.load_foundation_weights` resolves
warm-start weights local cache -> Workbench S3 -> public origin URL. This script
populates the S3 rung so training jobs never depend on the public internet: a
Zenodo ``HTTP 504`` killed ``pxr-reg-chemprop-chemeleon-phase1-frz20`` 14 seconds
into fold 1 on 2026-07-29, after frz0 and frz10 had already trained fine.

The download is **not** done here — fetch the checkpoint yourself, then point this
script at the local file::

    curl -O https://zenodo.org/records/15460715/files/chemeleon_mp.pt
    python scripts/admin/push_chemeleon_models.py --file chemeleon_mp.pt

    # inspect + hash, no upload
    python scripts/admin/push_chemeleon_models.py --file chemeleon_mp.pt --dry-run

    # a different registered checkpoint
    python scripts/admin/push_chemeleon_models.py --model chemeleon --file /tmp/chemeleon_mp.pt

Two objects are written per checkpoint::

    s3://$WORKBENCH_BUCKET/foundation-models/chemeleon/15460715/chemeleon_mp.pt
    s3://$WORKBENCH_BUCKET/foundation-models/chemeleon/15460715/SOURCE.json

The upload refuses to clobber an existing object unless ``--force`` is given, and
verifies the file really is a chemprop-style checkpoint first (``hyper_parameters``
+ ``state_dict``), so a truncated or HTML-error-page download can't get staged.
The file's md5 and byte size are also checked against the registry's expected values,
so a truncated download fails at the gate. Torch is optional here: without it the
structural check is skipped with a warning (the md5/size check still runs).
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from workbench.training.foundation_models import (
    foundation_entry,
    known_foundation_models,
    workbench_bucket,
)

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def hash_of(path: Path, algorithm: str = "sha256") -> str:
    """Streaming hash of a file (these checkpoints are tens of MB).

    Args:
        path (Path): File to hash.
        algorithm (str, optional): Any :mod:`hashlib` name. Defaults to "sha256".

    Returns:
        str: Hex digest.
    """
    digest = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_integrity(path: Path, entry: dict) -> dict:
    """Compare size and md5 against the registry's expected values.

    A truncated download, a captive-portal HTML page, or a checkpoint that changed
    under a stable record id all get caught here -- at the staging gate, once,
    rather than inside a training job. Mismatch raises; absent expectations warn.

    Args:
        path (Path): Local checkpoint file.
        entry (dict): Registry entry for this foundation model.

    Returns:
        dict: {"md5", "size_bytes", "integrity"} for the sidecar.

    Raises:
        ValueError: If size or md5 disagrees with the registry.
    """
    size = path.stat().st_size
    md5 = hash_of(path, "md5")
    expected_size = entry.get("expected_size_bytes")
    expected_md5 = entry.get("expected_md5")

    print(f"  md5:    {md5}")
    if expected_md5 is None and expected_size is None:
        log.warning("  No expected md5/size in the registry -- integrity unverified")
        return {"md5": md5, "size_bytes": size, "integrity": "unverified"}

    problems = []
    if expected_size is not None and size != expected_size:
        problems.append(f"size {size} != expected {expected_size}")
    if expected_md5 is not None and md5 != expected_md5:
        problems.append(f"md5 {md5} != expected {expected_md5}")
    if problems:
        raise ValueError(
            "Checkpoint integrity check FAILED: "
            + "; ".join(problems)
            + f". Re-download from {entry['origin_url']}; if it still mismatches, the origin "
            "may have published new weights -- add a new registry entry rather than staging over this one."
        )

    print(f"  integrity OK: matches expected md5 and size ({size} bytes)")
    return {"md5": md5, "size_bytes": size, "integrity": "verified"}


def verify_checkpoint(path: Path) -> dict:
    """Confirm the file loads as a chemprop-style MPNN checkpoint.

    Args:
        path (Path): Local checkpoint file.

    Returns:
        dict: Details for the sidecar ({"hidden_dim", "depth"} when known).
    """
    try:
        import torch
    except ImportError:
        log.warning("torch not installed — skipping the structural check (hash + upload only)")
        return {}

    try:
        ckpt = torch.load(path, weights_only=True)
    except Exception as e:
        raise ValueError(
            f"{path} does not load as a torch checkpoint ({type(e).__name__}: {e}). "
            "A truncated download or an HTML error page will look like this."
        ) from None
    missing = [k for k in ("hyper_parameters", "state_dict") if k not in ckpt]
    if missing:
        raise ValueError(f"{path} is not a chemprop foundation checkpoint (missing {missing})")
    hp = ckpt["hyper_parameters"]
    print(f"  checkpoint OK: hyper_parameters={hp}")
    return {"hyper_parameters": {k: v for k, v in hp.items() if isinstance(v, (int, float, str, bool))}}


def object_exists(client, bucket: str, key: str) -> bool:
    """True if s3://bucket/key is already there."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def push(model: str, local_file: Path, bucket: str, dry_run: bool, force: bool) -> int:
    """Verify and upload one checkpoint plus its SOURCE.json sidecar."""
    entry = foundation_entry(model)
    key = entry["s3_key"]
    sidecar_key = f"{key.rsplit('/', 1)[0]}/SOURCE.json"

    if not local_file.exists():
        log.error(f"No such file: {local_file}")
        return 1

    print(f"\n=== {model} ===")
    print(f"  local:  {local_file} ({local_file.stat().st_size / 1e6:.1f} MB)")
    print(f"  target: s3://{bucket}/{key}")

    integrity = check_integrity(local_file, entry)
    details = verify_checkpoint(local_file)
    checksum = hash_of(local_file, "sha256")
    print(f"  sha256: {checksum}")

    sidecar = {
        "model": model,
        "filename": entry["filename"],
        "origin_url": entry["origin_url"],
        "provenance_id": entry["provenance_id"],
        "description": entry["description"],
        "sha256": checksum,
        **integrity,
        "uploaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "uploaded_by": "scripts/admin/push_chemeleon_models.py",
        **details,
    }

    if dry_run:
        print("  DRY RUN — nothing uploaded. Sidecar would be:")
        print(json.dumps(sidecar, indent=2))
        return 0

    client = boto3.client("s3")
    if object_exists(client, bucket, key) and not force:
        log.warning(f"  s3://{bucket}/{key} already exists — use --force to overwrite. Skipping.")
        return 0

    client.upload_file(str(local_file), bucket, key)
    print(f"  uploaded {key}")
    client.put_object(
        Bucket=bucket,
        Key=sidecar_key,
        Body=json.dumps(sidecar, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"  uploaded {sidecar_key}")
    print(f"\n  Training jobs will now resolve '{model}' from S3 instead of {entry['origin_url']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--file", required=True, type=Path, help="Local checkpoint file to upload")
    parser.add_argument(
        "--model",
        default="chemeleon",
        choices=known_foundation_models(),
        help="Registered foundation model name (default: chemeleon)",
    )
    parser.add_argument("--bucket", default=None, help="Override the Workbench bucket")
    parser.add_argument("--dry-run", action="store_true", help="Verify and hash, but do not upload")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing S3 object")
    args = parser.parse_args()

    bucket = args.bucket or workbench_bucket()
    if not bucket:
        log.error("No WORKBENCH_BUCKET in the environment or Workbench config — pass --bucket")
        return 1

    try:
        return push(args.model, args.file, bucket, args.dry_run, args.force)
    except ValueError as e:
        log.error(f"\n  REFUSING TO STAGE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
