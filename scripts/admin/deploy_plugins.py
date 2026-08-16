"""Deploy Workbench's plugin pages to S3 and restart the dashboard.

Publishes to ``s3://<WORKBENCH_BUCKET>/workbench_plugins``, which is where a dashboard loads
plugins from when its ``WORKBENCH_PLUGINS`` points at that prefix. Deploying to S3 rather
than a local path is how a running dashboard (ECS, or anything without the repo checked out)
picks them up.

``--bump`` restarts the dashboard, which needs admin credentials. They have to come from an
admin ``WORKBENCH_CONFIG``: ConfigManager overwrites ``AWS_PROFILE`` from the config file, so
exporting an admin profile alongside a non-admin config silently runs as the wrong role.

Usage::

    python scripts/admin/deploy_plugins.py --dry-run --prune --bump
    python scripts/admin/deploy_plugins.py --prune --bump
    python scripts/admin/deploy_plugins.py --source ~/my_workbench_plugins
"""

import argparse
import os
import pathlib

import boto3

import workbench
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import copy_local_files_to_s3

# The prefix a dashboard's WORKBENCH_PLUGINS points at
PLUGIN_PREFIX = "workbench_plugins"

# The dashboard's ECS cluster and service are named by the CDK stack: a fixed logical id plus
# a per-account random suffix. Matching the stable part finds them in any account, which beats
# hardcoding names that differ everywhere.
CLUSTER_MATCH = "WorkbenchDashboard-WorkbenchCluster"
SERVICE_MATCH = "WorkbenchDashboard-WorkbenchService"


def files_to_deploy(source: pathlib.Path) -> list[pathlib.Path]:
    """Every file that would be uploaded, mirroring copy_local_files_to_s3's own filtering.

    Args:
        source (pathlib.Path): The plugin directory to deploy.

    Returns:
        list[pathlib.Path]: Paths relative to ``source``, sorted.
    """
    found = []
    for root, dirs, names in os.walk(source):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        found += [pathlib.Path(root, n).relative_to(source) for n in names if not n.endswith(".pyc")]
    return sorted(found)


def stale_keys(bucket: str, expected: set) -> list[str]:
    """Object keys under the plugin prefix that the source no longer has."""
    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    live = (
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{PLUGIN_PREFIX}/")
        for obj in page.get("Contents", [])
    )
    return sorted(key for key in live if key not in expected)


def bump_dashboard(wait: bool):
    """Force a new ECS deployment so the dashboard reloads plugins.

    Plugins are read once when a task starts, so an upload alone changes nothing until the
    service happens to cycle.

    Args:
        wait (bool): Block until the replacement task is serving.
    """
    ecs = boto3.client("ecs")
    clusters = [a for a in ecs.list_clusters()["clusterArns"] if CLUSTER_MATCH in a]
    if len(clusters) != 1:
        raise SystemExit(f"Expected one cluster matching {CLUSTER_MATCH!r}, found: {clusters}")
    services = [a for a in ecs.list_services(cluster=clusters[0])["serviceArns"] if SERVICE_MATCH in a]
    if len(services) != 1:
        raise SystemExit(f"Expected one service matching {SERVICE_MATCH!r}, found: {services}")

    ecs.update_service(cluster=clusters[0], service=services[0], forceNewDeployment=True)
    print(f"\nRestarting {services[0].rsplit('/', 1)[-1]}")
    if not wait:
        print("Live once the replacement task passes its health checks.")
        return

    print("Waiting for the service to stabilize (5-10 min; Ctrl-C is safe, the deploy continues).")
    ecs.get_waiter("services_stable").wait(cluster=clusters[0], services=services)
    print("Dashboard is back up with the new plugins.")


def main():
    ap = argparse.ArgumentParser(description="Deploy Workbench plugin pages to S3.")
    ap.add_argument("--source", type=pathlib.Path, help="Plugin directory. Defaults to Workbench's own.")
    ap.add_argument("--prune", action="store_true", help="Delete objects the source no longer has.")
    ap.add_argument("--bump", action="store_true", help="Restart the dashboard. Needs an admin WORKBENCH_CONFIG.")
    ap.add_argument("--no-wait", action="store_true", help="With --bump, don't wait for the service to stabilize.")
    ap.add_argument("--dry-run", action="store_true", help="List what would change and exit.")
    args = ap.parse_args()

    source = args.source or pathlib.Path(workbench.__file__).parent / "plugin_pages"
    if not source.is_dir():
        raise SystemExit(f"Source is not a directory: {source}")

    bucket = ConfigManager().get_config("WORKBENCH_BUCKET")
    if not bucket:
        raise SystemExit("WORKBENCH_BUCKET is not set; check WORKBENCH_CONFIG.")

    files = files_to_deploy(source)
    if not files:
        raise SystemExit(f"Nothing to deploy, no files under: {source}")

    print(f"source:      {source}")
    print(f"destination: s3://{bucket}/{PLUGIN_PREFIX}")
    for relative in files:
        print(f"  upload  {relative}")

    # Keys built exactly as copy_local_files_to_s3 builds them, so the two always agree
    stale = stale_keys(bucket, {f"{PLUGIN_PREFIX}/{f}" for f in files}) if args.prune else []
    for key in stale:
        print(f"  DELETE  {key}")

    if args.dry_run:
        print(f"\nDry run: {len(files)} upload(s), {len(stale)} delete(s){', then a restart' if args.bump else ''}.")
        return

    # Uploads overwrite by key; without --prune, files removed from the source stay in S3
    # and the dashboard keeps loading them.
    copy_local_files_to_s3(str(source), f"s3://{bucket}/{PLUGIN_PREFIX}")
    if stale:
        boto3.client("s3").delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in stale]})
    print(f"\nDeployed {len(files)} file(s), deleted {len(stale)}.")

    if args.bump:
        bump_dashboard(wait=not args.no_wait)
    else:
        print("The dashboard reads plugins at startup, so pass --bump to restart it.")


if __name__ == "__main__":
    main()
