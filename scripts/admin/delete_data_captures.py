"""Delete SageMaker data capture files for endpoints that have capture enabled.

EndpointCore.delete() deliberately preserves data_capture/, so capture files survive
endpoint deletion and accumulate across endpoint generations. This clears them out.

Dry run by default -- it reports what it would remove and stops. Pass --delete to
actually remove the files.

Usage:
    python scripts/admin/delete_data_captures.py
    python scripts/admin/delete_data_captures.py --before 2026-05-07
    python scripts/admin/delete_data_captures.py --endpoints herg-ic50-reg-1 --delete
"""

import argparse
import re
from datetime import datetime

import awswrangler as wr

from workbench.api import Endpoint
from workbench.cached.cached_meta import CachedMeta
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

# Capture keys are date-partitioned: .../yyyy/mm/dd/HH/MM-SS-<id>.jsonl
DATE_IN_KEY = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")


def capture_endpoints() -> list:
    """Names of endpoints that currently have data capture enabled."""
    df = CachedMeta().endpoints(details=True)
    return df[df["Capture"] == "True"]["Name"].tolist()


def list_capture_files(s3_client, path: str) -> list:
    """Return (uri, size, date) for every object under an S3 path. Date is None if the
    key isn't date-partitioned."""
    bucket, _, prefix = path.replace("s3://", "").partition("/")
    files = []
    for page in s3_client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            match = DATE_IN_KEY.search(obj["Key"])
            date = datetime(*map(int, match.groups())).date() if match else None
            files.append((f"s3://{bucket}/{obj['Key']}", obj["Size"], date))
    return files


def select(files: list, before, after) -> tuple:
    """Split files into (selected, kept) using the date window. Files whose key carries
    no date are always kept -- they can't be placed in the window."""
    if before is None and after is None:
        return files, []

    selected, kept = [], []
    for entry in files:
        date = entry[2]
        in_window = date is not None and (before is None or date < before) and (after is None or date >= after)
        (selected if in_window else kept).append(entry)
    return selected, kept


def main(endpoints: list, before, after, do_delete: bool):
    s3_client = AWSAccountClamp().boto3_session.client("s3")
    names = endpoints or capture_endpoints()
    print(f"{len(names)} endpoint(s) with data capture: {names}\n")

    all_selected = []
    for name in names:
        path = Endpoint(name).data_capture().data_capture_path
        files = list_capture_files(s3_client, path)
        if not files:
            print(f"{name}: no capture files")
            continue

        selected, kept = select(files, before, after)
        dates = [d for _, _, d in files if d]
        span = f"{min(dates)} -> {max(dates)}" if dates else "no dates in keys"
        print(f"{name}: {len(files)} files, {sum(s for _, s, _ in files) / 1e9:.2f} GB, {span}")
        print(f"    delete {len(selected)} ({sum(s for _, s, _ in selected) / 1e9:.2f} GB), keep {len(kept)}")
        all_selected.extend(uri for uri, _, _ in selected)

    print(f"\nTotal selected for deletion: {len(all_selected)} files")
    if not all_selected:
        return

    if not do_delete:
        print("Dry run -- pass --delete to remove these files.")
        return

    print("Deleting...")
    wr.s3.delete_objects(all_selected, boto3_session=AWSAccountClamp().boto3_session)
    print(f"Deleted {len(all_selected)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoints", nargs="+", help="endpoint names (default: every endpoint with capture on)")
    parser.add_argument(
        "--before",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="only files before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--after",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="only files on/after this date (YYYY-MM-DD)",
    )
    parser.add_argument("--delete", action="store_true", help="actually delete (default is a dry run)")
    args = parser.parse_args()
    main(args.endpoints, args.before, args.after, args.delete)
