"""Fetch the OpenADMET PXR Challenge train/test data from HuggingFace and
push it to the Workbench public data S3 bucket (`workbench-public-data`).

Once uploaded, the files are accessible via:

    from workbench.api import PublicData
    df = PublicData().get("comp_chem/openadmet_pxr/pxr_train")

Source:
    https://huggingface.co/datasets/openadmet/pxr-challenge-train-test

Files (all CSV on HF, stored as parquet on S3):
    pxr-challenge_TRAIN.csv                          4,140 rows   primary pEC50 training
    pxr-challenge_TEST_BLINDED.csv                     513 rows   blinded activity test
    pxr-challenge_counter-assay_TRAIN.csv            2,860 rows   PXR-null counter-assay
    pxr-challenge_single_concentration_TRAIN.csv    21,000 rows   single-conc primary screen (pretrain)
    pxr-challenge_structure_TEST_BLINDED.csv            78 rows   structure-track fragments

Run with the AWS profile that owns the public bucket, e.g.:

    AWS_PROFILE=scp_sandbox_admin python load_data.py

Descriptions for each file are merged into the shared
`s3://workbench-public-data/comp_chem/descriptions.json` index
(manually, outside this script) so `PublicData().describe(...)` returns
useful metadata.
"""

import argparse
import logging
from io import BytesIO

import boto3
import pandas as pd

log = logging.getLogger("openadmet_pxr.load_data")

BUCKET = "workbench-public-data"
S3_PREFIX = "comp_chem/openadmet_pxr"
HF_REPO = "openadmet/pxr-challenge-train-test"

# Map: S3 basename (without extension) -> HF CSV filename
FILES = {
    "pxr_train": "pxr-challenge_TRAIN.csv",
    "pxr_test_blinded": "pxr-challenge_TEST_BLINDED.csv",
    "pxr_counter_assay_train": "pxr-challenge_counter-assay_TRAIN.csv",
    "pxr_single_concentration_train": "pxr-challenge_single_concentration_TRAIN.csv",
    "pxr_structure_test_blinded": "pxr-challenge_structure_TEST_BLINDED.csv",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + snake_case column names (matches the rest of open_admet).

    Strips parentheses/units, collapses runs of underscores, trims ends so a
    column like ``"pEC50 Std Error (-log10(molarity))"`` becomes ``pec50_std_error``.
    """
    df = df.copy()
    cols = df.columns.str.lower()
    # Strip any parenthesised unit/annotation (including nested parens)
    while cols.str.contains(r"\(").any():
        cols = cols.str.replace(r"\([^()]*\)", "", regex=True)
    # Replace non-alphanumeric with underscore, collapse, strip
    cols = cols.str.replace(r"[^a-z0-9]+", "_", regex=True).str.replace(r"_+", "_", regex=True).str.strip("_")
    df.columns = cols
    return df


def fetch_from_hf(filename: str) -> pd.DataFrame:
    url = f"hf://datasets/{HF_REPO}/{filename}"
    log.info(f"Fetching {url}")
    df = pd.read_csv(url)
    return normalize_columns(df)


def s3_key(basename: str) -> str:
    return f"{S3_PREFIX}/{basename}.parquet"


def key_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def upload_parquet(s3, df: pd.DataFrame, key: str) -> None:
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    log.info(f"Wrote s3://{BUCKET}/{key}  ({len(df):,} rows, {len(df.columns)} cols)")


def main(overwrite: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    s3 = boto3.client("s3", region_name="us-west-2")

    for basename, hf_name in FILES.items():
        key = s3_key(basename)
        if not overwrite and key_exists(s3, key):
            log.info(f"Skipping (already exists): s3://{BUCKET}/{key}")
            continue
        df = fetch_from_hf(hf_name)
        upload_parquet(s3, df, key)

    log.info("Done. Access later via:")
    log.info('    PublicData().get("comp_chem/openadmet_pxr/pxr_train")')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Re-upload even if the key exists.")
    args = parser.parse_args()
    main(overwrite=args.overwrite)
