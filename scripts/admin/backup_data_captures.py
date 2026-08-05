"""Copy endpoint data capture files to a backup prefix in the same bucket.

    endpoints/<endpoint>/data_capture/...  ->  data_capture_backups/<endpoint>/data_capture/...

Endpoint discovery and the date window come from delete_data_captures.py, so the same
arguments select the same files. Run this first, with matching arguments, to keep a copy
of whatever that script is about to remove.

Copies are server-side, so object count drives the runtime rather than total size. Files
already present in the backup at the same size are skipped, so an interrupted run resumes
where it left off.

Dry run by default -- it reports what it would copy and stops. Pass --copy to run it.

Usage:
    python scripts/admin/backup_data_captures.py
    python scripts/admin/backup_data_captures.py --after 2026-05-07 --copy
    python scripts/admin/backup_data_captures.py --endpoints herg-ic50-reg-1 --copy --threads 64
"""

import argparse
import time
from datetime import datetime

import awswrangler as wr
from delete_data_captures import CAPTURE_DIR, capture_files_by_endpoint, select

from workbench.core.artifacts.artifact import Artifact
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

SOURCE_PREFIX = "/endpoints/"
BACKUP_PREFIX = "/data_capture_backups/"
CHUNK_SIZE = 1000  # Objects per copy call, so progress is reported as the run proceeds


def backed_up_sizes(s3_client, target_path: str) -> dict:
    """Map object key -> size for whatever is already under an endpoint's backup path."""
    bucket, _, prefix = target_path.replace("s3://", "").partition("/")
    existing = {}
    for page in s3_client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            existing[obj["Key"]] = obj["Size"]
    return existing


def pending(files: list, source_path: str, target_path: str, existing: dict) -> list:
    """Files not already backed up at the same size, so a re-run resumes instead of recopying."""
    source_prefix = source_path.replace("s3://", "").partition("/")[2]
    target_prefix = target_path.replace("s3://", "").partition("/")[2]
    todo = []
    for uri, size, _ in files:
        key = uri.replace("s3://", "").partition("/")[2]
        target_key = target_prefix + key[len(source_prefix) :]
        if existing.get(target_key) != size:
            todo.append((uri, size))
    return todo


def main(endpoints: list, before, after, do_copy: bool, threads: int):
    boto3_session = AWSAccountClamp().boto3_session
    s3_client = boto3_session.client("s3")
    captures = capture_files_by_endpoint(s3_client)
    if endpoints:
        captures = {name: files for name, files in captures.items() if name in endpoints}
    if not captures:
        print("No data capture files found.")
        return
    print(f"{len(captures)} endpoint(s) with capture files\n")

    plan, skipped = [], 0
    for name, files in sorted(captures.items()):
        selected, _ = select(files, before, after)
        source_path = f"{Artifact.endpoints_s3_path}/{name}/{CAPTURE_DIR}"
        target_path = source_path.replace(SOURCE_PREFIX, BACKUP_PREFIX, 1)

        todo = pending(selected, source_path, target_path, backed_up_sizes(s3_client, target_path))
        skipped += len(selected) - len(todo)
        size = sum(s for _, s in todo)
        print(f"{name}: copy {len(todo)} of {len(files)} files ({size / 1e9:.2f} GB)")
        if todo:
            plan.append((name, source_path, target_path, [uri for uri, _ in todo]))

    total = sum(len(uris) for *_, uris in plan)
    print(f"\nTotal selected for backup: {total} files ({skipped} already backed up)")
    if not total:
        return

    if not do_copy:
        print(f"Dry run -- pass --copy to back these up. Target looks like:\n    {plan[0][2]}/")
        return

    start, done = time.time(), 0
    for name, source_path, target_path, uris in plan:
        print(f"Copying {len(uris)} files for {name} -> {target_path}/")
        for offset in range(0, len(uris), CHUNK_SIZE):
            wr.s3.copy_objects(
                paths=uris[offset : offset + CHUNK_SIZE],
                source_path=source_path,
                target_path=target_path,
                use_threads=threads,
                boto3_session=boto3_session,
            )
            done += len(uris[offset : offset + CHUNK_SIZE])
            rate = done / max(time.time() - start, 1e-6)
            eta = (total - done) / rate if rate else 0
            print(f"    {done}/{total} files  {rate:.0f}/s  eta {eta / 60:.1f} min")
    print(f"Backed up {total} files in {(time.time() - start) / 60:.1f} min.")


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
    parser.add_argument("--copy", action="store_true", help="actually copy (default is a dry run)")
    parser.add_argument("--threads", type=int, default=32, help="concurrent copy requests (default: 32)")
    args = parser.parse_args()
    main(args.endpoints, args.before, args.after, args.copy, args.threads)
