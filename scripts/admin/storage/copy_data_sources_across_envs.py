"""Copy DataSources between AWS/Workbench environments.

Two modes:

  download  Pull each DataSource to a local parquet file (+ tags sidecar).
            Point your credentials at the SOURCE env, then run this.

  upload    Recreate each DataSource from the local files.
            Point your credentials at a TARGET env, then run this.
            Repeat for each target env.

Examples:
  # Source env
  python copy_data_sources_across_envs.py download

  # Each target env (switch creds in between)
  python copy_data_sources_across_envs.py upload
"""

import argparse
import json
from pathlib import Path

from workbench.api import DataSource
from workbench.core.transforms.pandas_transforms import PandasToData

# The DataSources to copy
DATA_SOURCES = [
    "foo",
    "bar",
]


def download(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds_name in DATA_SOURCES:
        print(f"\nDownloading DataSource: {ds_name}")
        ds = DataSource(ds_name)

        # Full pull (query, not pull_dataframe, to avoid the 100k row cap)
        df = ds.query(f"SELECT * FROM {ds.table}")
        tags = ds.get_tags()

        parquet_path = out_dir / f"{ds_name}.parquet"
        tags_path = out_dir / f"{ds_name}.tags.json"
        df.to_parquet(parquet_path, index=False)
        tags_path.write_text(json.dumps(tags))

        print(f"  rows={len(df)} cols={len(df.columns)} tags={tags}")
        print(f"  wrote {parquet_path}")


def upload(in_dir: Path):
    for ds_name in DATA_SOURCES:
        parquet_path = in_dir / f"{ds_name}.parquet"
        tags_path = in_dir / f"{ds_name}.tags.json"
        if not parquet_path.exists():
            print(f"\nSKIP {ds_name}: {parquet_path} not found")
            continue

        print(f"\nUploading DataSource: {ds_name}")
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        tags = json.loads(tags_path.read_text()) if tags_path.exists() else [ds_name]

        df_to_data = PandasToData(ds_name)  # same name in the target env
        df_to_data.set_output_tags(tags)
        df_to_data.set_input(df)
        df_to_data.transform()

        # Verify row count round-trips
        new_ds = DataSource(ds_name)
        new_df = new_ds.query(f"SELECT * FROM {new_ds.table}")
        ok = len(new_df) == len(df)
        status = "OK" if ok else "MISMATCH"
        print(f"  {status}: local rows={len(df)} -> new rows={len(new_df)} tags={tags}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy DataSources between AWS/Workbench envs.")
    parser.add_argument("mode", choices=["download", "upload"], help="download from source env, upload to target env")
    parser.add_argument("--dir", default="./ds_export", help="Local directory for the exported files (default: ./ds_export)")
    args = parser.parse_args()

    work_dir = Path(args.dir)
    if args.mode == "download":
        download(work_dir)
    else:
        upload(work_dir)
