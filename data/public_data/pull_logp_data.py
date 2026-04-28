"""Pull and merge publicly available experimental LogP datasets into a unified CSV.

Sources (octanol-water partition coefficient, experimental measurements only):
  1. PHYSPROP / OPERA      (~4.2K compounds via EPA / NIEHS)
  2. GraphormerLogP (GLP)  (~42K compounds, multi-source curation by CIMM Kazan)

Output:
  output/logp/logp_all.csv              -- merged, deduplicated on canonical SMILES
  output/logp/logp_<source>.csv         -- per-source files
"""

import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd

from alignment_utils import run_alignment_checks
from pull_common import download, merge_and_deduplicate, standardize_df

log = logging.getLogger("workbench")

OUTPUT_DIR = Path(__file__).parent / "output" / "logp"
VALUE_NAME = "logp"
FILE_PREFIX = "logp"


def pull_opera_physprop() -> pd.DataFrame:
    """PHYSPROP / OPERA logP training data (~4.2K compounds).

    Downloads OPERA_Data.zip from the NIEHS/OPERA GitHub repository and
    extracts the LogP training set.
    """
    log.info("Pulling OPERA / PHYSPROP logP training data ...")

    url = "https://raw.githubusercontent.com/NIEHS/OPERA/master/OPERA_Data.zip"
    raw = download(url, desc="OPERA_Data.zip")

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        logp_files = [f for f in zf.namelist() if "logp" in f.lower() and (f.endswith(".csv") or f.endswith(".txt"))]
        if not logp_files:
            logp_files = [f for f in zf.namelist() if "TR_LogP" in f or "logP" in f]
        if not logp_files:
            all_files = [f for f in zf.namelist() if not f.endswith("/")]
            log.warning(f"OPERA zip contents (no logP found): {all_files[:20]}")
            return pd.DataFrame(columns=["smiles", "canon_smiles", VALUE_NAME, "source"])

        log.info(f"OPERA LogP files found: {logp_files}")
        for logp_file in sorted(logp_files):
            if not logp_file.endswith(".csv"):
                continue
            try:
                with zf.open(logp_file) as fh:
                    df = pd.read_csv(fh)
                    log.info(f"  {logp_file}: {len(df)} rows, columns={df.columns.tolist()[:8]}")
                    smiles_col = next(
                        (c for c in df.columns if "smiles" in c.lower() or "canonical" in c.lower()),
                        None,
                    )
                    logp_col = next(
                        (c for c in df.columns if "value" in c.lower() or "logp" in c.lower() or c.lower() == "exp"),
                        None,
                    )
                    if smiles_col and logp_col and len(df) > 100:
                        return standardize_df(
                            df,
                            smiles_col=smiles_col,
                            value_col=logp_col,
                            source="opera_physprop",
                            value_name=VALUE_NAME,
                        )
            except Exception as e:
                log.warning(f"Failed to read {logp_file}: {e}")

    log.warning("OPERA / PHYSPROP: could not extract logP data from zip")
    return pd.DataFrame(columns=["smiles", "canon_smiles", VALUE_NAME, "source"])


def pull_graphormer_logp() -> pd.DataFrame:
    """GraphormerLogP (GLP) dataset (~42K experimental logP values)."""
    log.info("Pulling GraphormerLogP (GLP) dataset ...")

    urls_to_try = [
        "https://raw.githubusercontent.com/cimm-kzn/GraphormerLogP/main/data/final_data/GLP_dataset_not_separate.csv",
        "https://raw.githubusercontent.com/cimm-kzn/GraphormerLogP/main/data/separate_data/GLP_dataset.csv",
    ]

    for url in urls_to_try:
        try:
            raw = download(url, desc="GraphormerLogP")
            df = pd.read_csv(io.BytesIO(raw))
            smiles_col = next((c for c in df.columns if "smiles" in c.lower()), df.columns[0])
            logp_col = next((c for c in df.columns if "logp" in c.lower() or "log_p" in c.lower()), df.columns[1])
            return standardize_df(
                df,
                smiles_col=smiles_col,
                value_col=logp_col,
                source="graphormer_logp",
                value_name=VALUE_NAME,
            )
        except Exception as e:
            log.warning(f"GraphormerLogP URL failed ({url}): {e}")
            continue

    try:
        zip_url = "https://github.com/cimm-kzn/GraphormerLogP/archive/refs/heads/main.zip"
        raw = download(zip_url, desc="GraphormerLogP repo zip")
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_files = [f for f in zf.namelist() if f.endswith(".csv") and "data" in f.lower()]
            for csv_file in csv_files:
                with zf.open(csv_file) as fh:
                    df = pd.read_csv(fh)
                    smiles_col = next((c for c in df.columns if "smiles" in c.lower()), df.columns[0])
                    logp_col = next(
                        (c for c in df.columns if "logp" in c.lower() or "log_p" in c.lower()), df.columns[1]
                    )
                    if len(df) > 1000:
                        return standardize_df(
                            df,
                            smiles_col=smiles_col,
                            value_col=logp_col,
                            source="graphormer_logp",
                            value_name=VALUE_NAME,
                        )
    except Exception as e:
        log.warning(f"GraphormerLogP zip fallback failed: {e}")

    log.warning("GraphormerLogP: could not retrieve data")
    return pd.DataFrame(columns=["smiles", "canon_smiles", VALUE_NAME, "source"])


SOURCES = {
    "opera": pull_opera_physprop,
    "graphormer": pull_graphormer_logp,
}


def main():
    parser = argparse.ArgumentParser(description="Pull and merge public LogP datasets")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(SOURCES.keys()),
        choices=list(SOURCES.keys()),
        help="Which sources to pull (default: all)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name in args.sources:
        try:
            df = SOURCES[name]()
            log.info(f"  {name}: {len(df)} rows")
            frames.append(df)
        except Exception:
            log.exception(f"Failed to pull {name}")

    if not frames:
        log.error("No data pulled from any source!")
        return

    merged = merge_and_deduplicate(frames, output_dir=args.output_dir, value_name=VALUE_NAME, file_prefix=FILE_PREFIX)
    out_path = args.output_dir / f"{FILE_PREFIX}_all.csv"
    merged.to_csv(out_path, index=False)
    log.info(f"Saved merged dataset -> {out_path}  ({len(merged)} unique compounds)")

    print("\n" + "=" * 60)
    print("LogP Data Pull Summary")
    print("=" * 60)
    for source, grp in pd.concat(frames, ignore_index=True).groupby("source"):
        print(f"  {source:<35s} {len(grp):>7,} compounds")
    print(f"  {'TOTAL (deduplicated)':<35s} {len(merged):>7,} compounds")
    print(f"\nOutput: {args.output_dir}")
    print("=" * 60)

    run_alignment_checks(
        df=merged,
        value_col=VALUE_NAME,
        std_col=f"{VALUE_NAME}_std",
        count_col=f"{VALUE_NAME}_count",
        sources_col="sources",
        assay_name="LogP",
        expected_range=(-5, 10),
        expected_mean=(0, 5),
    )


if __name__ == "__main__":
    main()
