"""Deploy the plugin pages that ship with Workbench to the config's S3 bucket.

Publishes to ``s3://<WORKBENCH_BUCKET>/workbench_plugins``, which is what a dashboard
loads when its ``WORKBENCH_PLUGINS`` config points at that prefix. Deploying to S3 rather
than a local path is how a running dashboard (ECS, or anything without the repo checked
out) picks up plugins.

The dashboard reads plugins at startup, so restart it afterwards to see changes.

Usage::

    WORKBENCH_CONFIG=/path/to/config.json \\
        python scripts/admin/deploy_plugins.py --dry-run

    WORKBENCH_CONFIG=/path/to/config.json \\
        python scripts/admin/deploy_plugins.py

    # A different source (your own plugin repo):
    python scripts/admin/deploy_plugins.py --source ~/my_workbench_plugins
"""

import argparse
import logging
import os
import pathlib

import workbench
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import copy_local_files_to_s3

log = logging.getLogger("workbench")

# The prefix a dashboard's WORKBENCH_PLUGINS points at
PLUGIN_PREFIX = "workbench_plugins"

# What gets uploaded, mirroring copy_local_files_to_s3's own filtering
SKIP_DIRS = {"__pycache__"}
SKIP_SUFFIXES = (".pyc",)


def packaged_plugins() -> pathlib.Path:
    """The plugin pages directory that ships inside the workbench package."""
    return pathlib.Path(workbench.__file__).parent / "plugin_pages"


def files_to_deploy(source: pathlib.Path) -> list[pathlib.Path]:
    """Every file that would be uploaded, in walk order.

    Args:
        source (pathlib.Path): The plugin directory to deploy.

    Returns:
        list[pathlib.Path]: Paths relative to ``source``.
    """
    found = []
    for root, dirs, names in os.walk(source):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(names):
            if name.endswith(SKIP_SUFFIXES):
                continue
            found.append(pathlib.Path(root, name).relative_to(source))
    return sorted(found)


def deployed_keys(bucket: str, session) -> list[str]:
    """Every object key currently under the plugin prefix.

    Args:
        bucket (str): The Workbench bucket.
        session: A boto3 session.

    Returns:
        list[str]: The keys, sorted.
    """
    paginator = session.client("s3").get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{PLUGIN_PREFIX}/"):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return sorted(keys)


def delete_keys(bucket: str, keys: list[str], session):
    """Delete objects, in the 1000-per-call batches the API allows."""
    client = session.client("s3")
    for start in range(0, len(keys), 1000):
        batch = keys[start : start + 1000]
        client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in batch]})
        log.important(f"Deleted {len(batch)} stale object(s) from {bucket}/{PLUGIN_PREFIX}")


def main():
    ap = argparse.ArgumentParser(description="Deploy Workbench plugin pages to S3.")
    ap.add_argument(
        "--source",
        type=pathlib.Path,
        help="Plugin directory to deploy. Defaults to the plugin_pages shipped with Workbench.",
    )
    ap.add_argument(
        "--prune",
        action="store_true",
        help="Delete objects under the prefix that aren't in the source. The dashboard loads "
        "whatever is there, so a file only removed locally keeps getting loaded without this.",
    )
    ap.add_argument("--dry-run", action="store_true", help="List what would change and exit.")
    args = ap.parse_args()

    source = args.source or packaged_plugins()
    if not source.is_dir():
        raise SystemExit(f"Source is not a directory: {source}")

    bucket = ConfigManager().get_config("WORKBENCH_BUCKET")
    if not bucket:
        raise SystemExit("WORKBENCH_BUCKET is not set; check WORKBENCH_CONFIG.")
    destination = f"s3://{bucket}/{PLUGIN_PREFIX}"

    files = files_to_deploy(source)
    if not files:
        raise SystemExit(f"Nothing to deploy, no files under: {source}")

    print(f"source:      {source}")
    print(f"destination: {destination}")
    for relative in files:
        print(f"  upload  {relative}")

    # Built the same way copy_local_files_to_s3 builds them, so the two always agree
    stale = []
    if args.prune:
        expected = {f"{PLUGIN_PREFIX}/{relative}" for relative in files}
        stale = [key for key in deployed_keys(bucket, AWSAccountClamp().boto3_session) if key not in expected]
        for key in stale:
            print(f"  DELETE  {key}")

    if args.dry_run:
        summary = f"\nDry run: {len(files)} file(s) would be uploaded"
        print(f"{summary}, {len(stale)} deleted." if args.prune else f"{summary}.")
        return

    # Uploads overwrite by key. Without --prune, files removed from the source stay in S3
    # and the dashboard keeps loading them.
    copy_local_files_to_s3(str(source), destination)
    if stale:
        delete_keys(bucket, stale, AWSAccountClamp().boto3_session)
    print(f"\nDeployed {len(files)} file(s), deleted {len(stale)}. Restart the dashboard to load them.")


if __name__ == "__main__":
    main()
