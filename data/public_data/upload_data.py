"""Maintainer-only: publish local public_data CSVs to s3://workbench-public-data.

Not for general/public use — write access to the public bucket is restricted
to SuperCowPowers maintainers. Have a dataset you'd like to see published?
We're happy to add additional public data — contact support@supercowpowers.com.

Walks `output/<subdir>/*.csv` and uploads each file to:
    s3://workbench-public-data/comp_chem/<subdir>/<filename>.csv

Then merges entries from the local `descriptions.json` into the top-level
`s3://workbench-public-data/descriptions.json` (read-merge-write — existing
remote entries for other datasets are preserved).

Defaults to a dry run that only prints what would happen. Pass --apply to
actually upload:

    AWS_PROFILE=scp_sandbox_admin python upload_data.py --apply
"""

import argparse
import json
import logging
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

log = logging.getLogger("workbench")

BUCKET = "workbench-public-data"
DESCRIPTIONS_KEY = "descriptions.json"  # top-level (matches PublicData._load_descriptions)
S3_PREFIX = "comp_chem"  # everything in this directory lives under comp_chem/

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "output"
LOCAL_DESCRIPTIONS = DATA_DIR / "descriptions.json"


def csv_files() -> list[Path]:
    """All CSVs under output/, sorted for deterministic output."""
    return sorted(OUTPUT_DIR.rglob("*.csv"))


def s3_key_for(local_path: Path) -> str:
    """Map output/<subdir>/<file>.csv -> comp_chem/<subdir>/<file>.csv."""
    rel = local_path.relative_to(OUTPUT_DIR)
    return f"{S3_PREFIX}/{rel.as_posix()}"


def upload_csvs(dry_run: bool) -> list[str]:
    """Upload every CSV under output/ and return the S3 keys written."""
    keys = []
    log.info("CSV uploads:")
    for path in csv_files():
        key = s3_key_for(path)
        df = pd.read_csv(path)
        s3_uri = f"s3://{BUCKET}/{key}"
        log.info(f"  {path.relative_to(DATA_DIR)} -> {s3_uri}  ({len(df):,} rows, {len(df.columns)} cols)")
        if not dry_run:
            wr.s3.to_csv(df, s3_uri, index=False)
        keys.append(key)
    if not keys:
        log.warning(f"No CSVs found under {OUTPUT_DIR}")
    return keys


def merge_descriptions() -> dict:
    """Merge local descriptions.json on top of the remote one. Local wins.

    Reads the remote with an unsigned client (the bucket is public-read), so
    this works without AWS credentials — only the put_object below needs them.
    """
    if not LOCAL_DESCRIPTIONS.exists():
        log.warning(f"No local descriptions.json at {LOCAL_DESCRIPTIONS}; skipping merge")
        return {}

    local = json.loads(LOCAL_DESCRIPTIONS.read_text())

    s3_anon = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))
    try:
        obj = s3_anon.get_object(Bucket=BUCKET, Key=DESCRIPTIONS_KEY)
        remote = json.loads(obj["Body"].read())
    except s3_anon.exceptions.NoSuchKey:
        remote = {}
    except s3_anon.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            remote = {}
        else:
            raise

    merged = {**remote, **local}
    return merged


def upload_descriptions(merged: dict, dry_run: bool) -> None:
    body = json.dumps(merged, indent=2).encode("utf-8")
    log.info(f"\ndescriptions.json: {len(merged)} entries -> s3://{BUCKET}/{DESCRIPTIONS_KEY}")
    for key in sorted(merged.keys()):
        log.info(f"  - {key}")
    if not dry_run:
        boto3.client("s3").put_object(
            Bucket=BUCKET,
            Key=DESCRIPTIONS_KEY,
            Body=body,
            ContentType="application/json",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually upload (default: dry run)")
    parser.add_argument(
        "--skip-descriptions", action="store_true", help="Only upload CSVs; do not touch descriptions.json"
    )
    parser.add_argument(
        "--skip-csvs", action="store_true", help="Only update descriptions.json; do not upload any CSVs"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    dry_run = not args.apply
    if dry_run:
        log.info("DRY RUN — pass --apply to actually upload\n")

    if not args.skip_csvs:
        upload_csvs(dry_run)

    if not args.skip_descriptions:
        merged = merge_descriptions()
        if merged:
            upload_descriptions(merged, dry_run)

    log.info("\nDone." if not dry_run else "\nDry run complete — re-run with --apply.")


if __name__ == "__main__":
    main()
