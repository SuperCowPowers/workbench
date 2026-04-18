"""Shared helpers for the reference-compounds populate scripts under scripts/admin/.

Each populate_*_reference_compounds.py defines a compound list and a description
dict, then calls run_populate() to upload the CSV and patch the shared
comp_chem/descriptions.json. Keeps the per-set scripts short and focused on
the compound definitions.

Not a public API — only imported by the populate scripts in this directory
(Python adds scripts/admin/ to sys.path when those scripts are invoked).
"""

import json
import logging

import awswrangler as wr
import boto3
import pandas as pd

log = logging.getLogger("workbench")

BUCKET = "workbench-public-data"
DESCRIPTIONS_KEY = "comp_chem/descriptions.json"


def s3_path_for(csv_key: str) -> str:
    return f"s3://{BUCKET}/{csv_key}"


def upload_csv(df: pd.DataFrame, csv_key: str) -> None:
    """Upload the reference DataFrame to S3 as a CSV."""
    path = s3_path_for(csv_key)
    log.info(f"Uploading CSV → {path}")
    wr.s3.to_csv(df, path, index=False)


def update_descriptions(csv_basename: str, description_dict: dict) -> None:
    """Patch a single entry into comp_chem/descriptions.json keyed by filename."""
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=DESCRIPTIONS_KEY)
        descriptions = json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        descriptions = {}

    descriptions[csv_basename] = description_dict

    body = json.dumps(descriptions, indent=2).encode("utf-8")
    log.info(f"Updating descriptions → s3://{BUCKET}/{DESCRIPTIONS_KEY}")
    s3.put_object(Bucket=BUCKET, Key=DESCRIPTIONS_KEY, Body=body, ContentType="application/json")


def run_populate(
    df: pd.DataFrame,
    csv_key: str,
    description_dict: dict,
    dry_run: bool = False,
) -> None:
    """Print the DataFrame, then upload + patch descriptions unless dry-run."""
    log.info(f"\nReference compounds ({len(df)} rows) for {csv_key}:")
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_colwidth", 40):
        log.info(df.to_string(index=False))

    if dry_run:
        log.info("\n[DRY RUN] Skipping S3 upload")
        return

    upload_csv(df, csv_key)
    csv_basename = csv_key.rsplit("/", 1)[-1]
    update_descriptions(csv_basename, description_dict)
    log.info("\nDone.")
