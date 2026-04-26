"""df_store: Pull/push DataFrames between a local cache and the AWS DFStore.

Used to migrate DataFrames between AWS environments. Pull from one
environment, switch WORKBENCH_CONFIG, then push to the other.

Usage:
    df_store pull <df_store_path>    Pull all DataFrames under <df_store_path>
                                     into ~/tmp/workbench/df_store
    df_store push                    Push everything under ~/tmp/workbench/df_store
                                     back to the DFStore at the same locations
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.fs as pa_fs
import pyarrow.parquet as pq

from workbench.api import DFStore

LOCAL_BASE = Path.home() / "tmp" / "workbench" / "df_store"


def location_to_local_path(location: str) -> Path:
    relative = location.lstrip("/")
    return LOCAL_BASE / f"{relative}.parquet"


def local_path_to_location(local_path: Path) -> str:
    relative = local_path.relative_to(LOCAL_BASE).with_suffix("")
    return "/" + relative.as_posix()


def _pa_s3_filesystem(boto3_session) -> pa_fs.S3FileSystem:
    creds = boto3_session.get_credentials().get_frozen_credentials()
    return pa_fs.S3FileSystem(
        access_key=creds.access_key,
        secret_key=creds.secret_key,
        session_token=creds.token,
        region=boto3_session.region_name,
    )


def get_shape(df_store: DFStore, fs: pa_fs.S3FileSystem, location: str) -> tuple:
    s3_uri = df_store._generate_s3_uri(location)
    path = s3_uri[len("s3://") :]
    dataset = pq.ParquetDataset(path, filesystem=fs)
    num_cols = len(dataset.schema.names)
    num_rows = sum(pq.ParquetFile(f, filesystem=fs).metadata.num_rows for f in dataset.files)
    return num_rows, num_cols


def pull(df_store_path: str):
    df_store = DFStore()
    locations = df_store.list(prefix=df_store_path)
    if not locations:
        print(f"No DataFrames found under '{df_store_path}'")
        return

    fs = _pa_s3_filesystem(df_store.boto3_session)
    print(f"Found {len(locations)} DataFrames under '{df_store_path}':")
    for location in locations:
        try:
            rows, cols = get_shape(df_store, fs, location)
            print(f"  {location}  ({rows} rows x {cols} cols)")
        except Exception as e:
            print(f"  {location}  (shape unavailable: {e})")
    print(f"\nAll existing files under {LOCAL_BASE} will be DELETED before pulling.")
    answer = input("Proceed? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        return

    if LOCAL_BASE.exists():
        shutil.rmtree(LOCAL_BASE)
    LOCAL_BASE.mkdir(parents=True, exist_ok=True)
    print(f"Pulling {len(locations)} DataFrames into {LOCAL_BASE}...")

    for location in locations:
        local_path = location_to_local_path(location)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df = df_store.get(location)
        if df is None:
            print(f"  SKIP {location} (not found)")
            continue
        df.to_parquet(local_path)
        print(f"  PULL {location} -> {local_path} ({len(df)} rows)")


def push():
    if not LOCAL_BASE.exists():
        print(f"Local store '{LOCAL_BASE}' does not exist")
        return

    files = sorted(LOCAL_BASE.rglob("*.parquet"))
    if not files:
        print(f"No .parquet files found in {LOCAL_BASE}")
        return

    df_store = DFStore()
    print(f"Pushing {len(files)} DataFrames from {LOCAL_BASE}...")

    for local_path in files:
        location = local_path_to_location(local_path)
        df = pd.read_parquet(local_path)
        df_store.upsert(location, df)
        print(f"  PUSH {local_path} -> {location} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Pull and push DataFrames between a local cache and the AWS DFStore")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    pull_parser = subparsers.add_parser("pull", help="Pull DataFrames from DFStore to local cache")
    pull_parser.add_argument("df_store_path", help="Path prefix in DFStore to pull recursively (e.g. /projects/foo)")

    subparsers.add_parser("push", help="Push DataFrames from local cache back to DFStore")

    args = parser.parse_args()

    if args.mode == "pull":
        pull(args.df_store_path)
    elif args.mode == "push":
        push()


if __name__ == "__main__":
    main()
