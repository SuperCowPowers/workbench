#!/usr/bin/env python3
import boto3
import argparse
from collections import defaultdict

# Workbench Imports
from workbench.utils.config_manager import ConfigManager


def list_and_delete_data_capture(bucket, preview=True):
    s3 = boto3.client("s3")
    prefix = "endpoints/"

    # Get all endpoint prefixes
    paginator = s3.get_paginator("list_objects_v2")
    endpoints = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            endpoint_prefix = common_prefix["Prefix"]
            if endpoint_prefix != prefix:
                endpoints.add(endpoint_prefix)

    # Count and optionally delete files per endpoint
    counts = defaultdict(int)
    to_delete = []

    for endpoint_prefix in endpoints:
        data_capture_prefix = f"{endpoint_prefix}data_capture/"

        for page in paginator.paginate(Bucket=bucket, Prefix=data_capture_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".jsonl"):
                    endpoint_name = endpoint_prefix.split("/")[-2]
                    counts[endpoint_name] += 1
                    to_delete.append(obj["Key"])

    # Display results
    if counts:
        print(f"\n{'Endpoint':<50} {'Files'}")
        print("-" * 60)
        for endpoint, count in sorted(counts.items()):
            print(f"{endpoint:<50} {count:>6}")
        print("-" * 60)
        print(f"{'TOTAL':<50} {sum(counts.values()):>6}\n")
    else:
        print("No .jsonl files found in data_capture directories")
        return

    # Delete if not preview mode
    if not preview:
        print("Deleting files...")
        # Delete in batches of 1000
        for i in range(0, len(to_delete), 1000):
            batch = to_delete[i : i + 1000]
            s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in batch]})
        print(f"Deleted {len(to_delete)} files")
    else:
        print("PREVIEW MODE - no files deleted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove data capture files from SageMaker endpoints")
    parser.add_argument("--delete", action="store_true", help="Actually delete files (default is preview)")
    args = parser.parse_args()

    # Get the default S3 bucket from Workbench config
    config = ConfigManager()
    bucket = config.get_config("WORKBENCH_BUCKET")
    print(f"Using S3 bucket: {bucket}")

    list_and_delete_data_capture(bucket, preview=not args.delete)
