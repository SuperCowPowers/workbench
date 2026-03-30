"""Pull and merge publicly available LogP datasets into a unified CSV.

Sources (experimental logP only):
  1. PHYSPROP / OPERA                (~14K experimental logP via EPA CompTox)
  2. GraphormerLogP (GLP)            (~42K experimental logP)

Output:
  output/logp_all.csv              -- merged, deduplicated on canonical SMILES
  output/logp_<source>.csv         -- per-source files
"""

import argparse
import io
import zipfile
from pathlib import Path
import logging
import pandas as pd
import requests
from rdkit import Chem
from tqdm import tqdm

# Workbench imports
from workbench.utils.chem_utils.mol_standardize import MolStandardizer

log = logging.getLogger("workbench")


OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shared standardizer instance -- full ChEMBL pipeline with mixture rejection enabled
_standardizer = MolStandardizer(canonicalize_tautomer=True, remove_salts=True, drop_mixtures=True)


def standardize_smiles(smiles: str) -> str | None:
    """Standardize a SMILES string using the workbench MolStandardizer.

    Applies: cleanup -> salt removal -> charge neutralization -> tautomer canonicalization.
    Multi-component entries with unknown large fragments are rejected (drop_invalid=True).

    Returns canonical SMILES of the parent molecule, or None if invalid/rejected.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    std_mol, _salt = _standardizer.standardize(mol)
    if std_mol is None:
        return None
    return Chem.MolToSmiles(std_mol, canonical=True)


def download(url: str, desc: str = "") -> bytes:
    """Download a URL with a progress bar and return bytes."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=desc or url.split("/")[-1]) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            buf.write(chunk)
            pbar.update(len(chunk))
    return buf.getvalue()


def standardize_df(df: pd.DataFrame, smiles_col: str, value_col: str, source: str) -> pd.DataFrame:
    """Standardize a dataframe to (smiles, canon_smiles, logp, source)."""
    out = df[[smiles_col, value_col]].copy()
    out.columns = ["smiles", "logp"]
    out["logp"] = pd.to_numeric(out["logp"], errors="coerce")
    out.dropna(subset=["logp"], inplace=True)
    out["canon_smiles"] = out["smiles"].apply(standardize_smiles)
    out.dropna(subset=["canon_smiles"], inplace=True)
    out["source"] = source
    return out[["smiles", "canon_smiles", "logp", "source"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source pullers
# ---------------------------------------------------------------------------


def pull_opera_physprop() -> pd.DataFrame:
    """PHYSPROP / OPERA logP training data (~14K compounds).

    Downloads OPERA_Data.zip from the NIEHS/OPERA GitHub repository and
    extracts the LogP training set.
    """
    log.info("Pulling OPERA / PHYSPROP logP training data ...")

    url = "https://raw.githubusercontent.com/NIEHS/OPERA/master/OPERA_Data.zip"
    raw = download(url, desc="OPERA_Data.zip")

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Find the LogP training file inside the zip
        logp_files = [f for f in zf.namelist() if "logp" in f.lower() and (f.endswith(".csv") or f.endswith(".txt"))]
        if not logp_files:
            # Try broader search for any training data files
            logp_files = [f for f in zf.namelist() if "TR_LogP" in f or "logP" in f]
        if not logp_files:
            # List what's available so we can debug
            all_files = [f for f in zf.namelist() if not f.endswith("/")]
            log.warning(f"OPERA zip contents (no logP found): {all_files[:20]}")
            return pd.DataFrame(columns=["smiles", "canon_smiles", "logp", "source"])

        log.info(f"OPERA LogP files found: {logp_files}")
        for logp_file in sorted(logp_files):
            if not logp_file.endswith(".csv"):
                continue
            try:
                with zf.open(logp_file) as fh:
                    df = pd.read_csv(fh)
                    log.info(f"  {logp_file}: {len(df)} rows, columns={df.columns.tolist()[:8]}")
                    # Column names: Original_SMILES, Canonical_QSARr, value_point_estimate
                    smiles_col = next(
                        (c for c in df.columns if "smiles" in c.lower() or "canonical" in c.lower()),
                        None,
                    )
                    logp_col = next(
                        (c for c in df.columns if "value" in c.lower() or "logp" in c.lower() or c.lower() == "exp"),
                        None,
                    )
                    if smiles_col and logp_col and len(df) > 100:
                        return standardize_df(df, smiles_col=smiles_col, value_col=logp_col, source="opera_physprop")
            except Exception as e:
                log.warning(f"Failed to read {logp_file}: {e}")

    log.warning("OPERA / PHYSPROP: could not extract logP data from zip")
    return pd.DataFrame(columns=["smiles", "canon_smiles", "logp", "source"])


def pull_graphormer_logp() -> pd.DataFrame:
    """GraphormerLogP (GLP) dataset (~42K experimental logP values).

    Downloads from the GraphormerLogP GitHub repository.
    """
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
            return standardize_df(df, smiles_col=smiles_col, value_col=logp_col, source="graphormer_logp")
        except Exception as e:
            log.warning(f"GraphormerLogP URL failed ({url}): {e}")
            continue

    # Try the repo zip as a fallback
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
                    if len(df) > 1000:  # sanity check -- we want the big dataset
                        return standardize_df(df, smiles_col=smiles_col, value_col=logp_col, source="graphormer_logp")
    except Exception as e:
        log.warning(f"GraphormerLogP zip fallback failed: {e}")

    log.warning("GraphormerLogP: could not retrieve data")
    return pd.DataFrame(columns=["smiles", "canon_smiles", "logp", "source"])


# ---------------------------------------------------------------------------
# Merge & deduplicate
# ---------------------------------------------------------------------------


def merge_and_deduplicate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge all frames and deduplicate on canonical SMILES.

    When the same compound appears in multiple sources we keep all rows
    (useful for cross-source comparison) but also flag duplicates.
    The deduplicated version takes the mean logP across sources.
    """
    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total rows before dedup: {len(combined)}")

    # Per-source file saves (before dedup)
    for source, grp in combined.groupby("source"):
        path = OUTPUT_DIR / f"logp_{source}.csv"
        grp.to_csv(path, index=False)
        log.info(f"  Saved {len(grp):>6,} rows -> {path.name}")

    # Deduplicate: group by canon_smiles, take mean logP, track sources
    dedup = (
        combined.groupby("canon_smiles")
        .agg(
            logp_mean=("logp", "mean"),
            logp_std=("logp", "std"),
            logp_count=("logp", "count"),
            sources=("source", lambda x: "|".join(sorted(set(x)))),
            logp_values=("logp", lambda x: "|".join(f"{v:.3f}" for v in x)),
        )
        .reset_index()
    )
    dedup["logp_std"] = dedup["logp_std"].fillna(0.0)

    # Rename/reorder for downstream consumers:
    #   id, smiles, logp  (the easy-to-use columns)
    #   logp_std, logp_count, sources, logp_values  (provenance/detail)
    dedup = dedup.rename(columns={"canon_smiles": "smiles", "logp_mean": "logp"})
    dedup.insert(0, "id", range(len(dedup)))
    dedup = dedup[["id", "smiles", "logp", "logp_std", "logp_count", "sources", "logp_values"]]

    log.info(f"Unique compounds after dedup: {len(dedup)}")
    return dedup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SOURCES = {
    "opera": pull_opera_physprop,
    "graphormer": pull_graphormer_logp,
}


def _set_output_dir(path: Path):
    global OUTPUT_DIR
    OUTPUT_DIR = path


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
    _set_output_dir(args.output_dir)

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

    merged = merge_and_deduplicate(frames)
    out_path = OUTPUT_DIR / "logp_all.csv"
    merged.to_csv(out_path, index=False)
    log.info(f"Saved merged dataset -> {out_path}  ({len(merged)} unique compounds)")

    # Print summary
    print("\n" + "=" * 60)
    print("LogP Data Pull Summary")
    print("=" * 60)
    for source, grp in pd.concat(frames, ignore_index=True).groupby("source"):
        print(f"  {source:<35s} {len(grp):>7,} compounds")
    print(f"  {'TOTAL (deduplicated)':<35s} {len(merged):>7,} compounds")
    print(f"\nOutput: {OUTPUT_DIR}")
    print("=" * 60)

    # Run alignment checks on the merged dataset
    from alignment_utils import run_alignment_checks

    run_alignment_checks(
        df=merged,
        value_col="logp",
        std_col="logp_std",
        count_col="logp_count",
        sources_col="sources",
        assay_name="LogP",
        expected_range=(-5, 10),
        expected_mean=(0, 5),
    )


if __name__ == "__main__":
    main()
