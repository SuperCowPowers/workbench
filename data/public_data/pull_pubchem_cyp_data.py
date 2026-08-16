"""Pull the Veith CYP inhibition panel (PubChem AID 1851) into public-data CSVs.

A qHTS 15-point dose-response screen of ~17k compounds against five human CYP
isoforms (1A2, 2C9, 2C19, 2D6, 3A4), reported one row per compound-isoform pair.
Structures ship with the assay, so no CID lookup is needed.

The public ML datasets derived from this screen (TDC, MoleculeNet) binarize it at
10 uM. This pull keeps the fitted AC50 as a pIC50 plus the qHTS curve-quality
columns, which is what regression against a dose-response target needs.

Writes `output/comp_chem/pubchem/cyp_inhibition/` -- one long file over all
isoforms plus a per-isoform file. The shared `upload_data.py` then publishes them
and merges the matching `descriptions.json` entries.

Run:
    python pull_pubchem_cyp_data.py
    AWS_PROFILE=scp_sandbox_admin python upload_data.py --apply   # then publish

On each run this prints a per-file row count, column list, and a paste-ready
`columns` skeleton to help keep `descriptions.json` in sync with the real schema.
"""

import argparse
import io
import json
import logging
from pathlib import Path

import pandas as pd
import requests
from pull_common import standardize_smiles

log = logging.getLogger("workbench")

OUTPUT_DIR = Path(__file__).parent / "output" / "comp_chem" / "pubchem" / "cyp_inhibition"

AID = 1851
REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
SID_CHUNK = 8000  # PUG-REST rejects assay requests over 10k SIDs

# Assay "Panel Name" -> isoform slug used in the output filenames and columns
ISOFORMS = {
    "p450-cyp1a2": "cyp1a2",
    "p450-cyp2c9": "cyp2c9",
    "p450-cyp2c19": "cyp2c19",
    "p450-cyp2d6": "cyp2d6",
    "p450-cyp3a4": "cyp3a4",
}

# Source column -> output column. Fit_LogAC50 is log10(AC50) in molar, so pIC50 is
# its negation; Potency is the same value in micromolar and is kept as reported.
COLUMNS = {
    "PUBCHEM_SID": "sid",
    "PUBCHEM_CID": "cid",
    "PUBCHEM_EXT_DATASOURCE_SMILES": "smiles",  # renamed to smiles_orig once standardized
    "PUBCHEM_ACTIVITY_OUTCOME": "activity_outcome",
    "Panel Name": "isoform",
    "Potency": "ac50_um",
    "Fit_LogAC50": "fit_log_ac50",
    "Fit_CurveClass": "curve_class",
    "Fit_R2": "fit_r2",
    "Fit_HillSlope": "hill_slope",
    "Max_Response": "max_response",
}


def fetch_sids() -> list[str]:
    """Every substance ID assayed in AID 1851."""
    resp = requests.get(f"{REST}/assay/aid/{AID}/sids/TXT", timeout=120)
    resp.raise_for_status()
    sids = resp.text.split()
    log.info(f"AID {AID}: {len(sids):,} SIDs")
    return sids


def fetch_chunk(sids: list[str]) -> pd.DataFrame:
    """Assay rows for one chunk of SIDs. POSTed because the SID list outgrows a URL."""
    resp = requests.post(f"{REST}/assay/aid/{AID}/CSV", data={"sid": ",".join(sids)}, timeout=600)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), low_memory=False)


def pull_cyp_panel() -> dict[str, pd.DataFrame]:
    """Fetch the panel, then split it into a long table plus one table per isoform."""
    sids = fetch_sids()
    chunks = [sids[i : i + SID_CHUNK] for i in range(0, len(sids), SID_CHUNK)]
    frames = []
    for n, chunk in enumerate(chunks, start=1):
        log.info(f"Fetching chunk {n}/{len(chunks)} ({len(chunk):,} SIDs)")
        frames.append(fetch_chunk(chunk))
    raw = pd.concat(frames, ignore_index=True)

    # PubChem prefixes the data with type/description/unit rows, which carry no SID
    raw = raw[raw["PUBCHEM_SID"].notna()]

    df = raw[list(COLUMNS)].rename(columns=COLUMNS)
    df["isoform"] = df["isoform"].map(ISOFORMS)
    df = df[df["isoform"].notna()]

    for col in ["sid", "cid"]:
        df[col] = df[col].astype("Int64")
    for col in ["ac50_um", "fit_log_ac50", "curve_class", "fit_r2", "hill_slope", "max_response"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.insert(5, "pic50", -df["fit_log_ac50"])

    # Deposited structures include salts and mixtures. `smiles` holds the standardized
    # form used for modeling; the deposited string is kept as `smiles_orig`.
    df = df.rename(columns={"smiles": "smiles_orig"})
    structures = df[["smiles_orig"]].drop_duplicates()
    structures["smiles"] = structures["smiles_orig"].apply(standardize_smiles)
    df = df.merge(structures, on="smiles_orig", how="left")
    dropped = df["smiles"].isna().sum()
    if dropped:
        log.warning(f"Dropping {dropped:,} rows whose SMILES could not be standardized")
        df = df[df["smiles"].notna()]
    df.insert(2, "smiles", df.pop("smiles"))

    df = df.sort_values(["sid", "isoform"]).reset_index(drop=True)

    out = {"all_isoforms": df}
    for isoform in sorted(ISOFORMS.values()):
        out[isoform] = df[df["isoform"] == isoform].drop(columns=["isoform"]).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Pull the Veith CYP inhibition panel from PubChem")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    skeleton = {}
    print("\n" + "=" * 70)
    print("PubChem CYP Panel Pull (AID 1851)")
    print("=" * 70)
    for name, df in pull_cyp_panel().items():
        out_path = args.output_dir / f"{name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        measured = int(df["pic50"].notna().sum())
        log.info(f"  {name}.csv  ({len(df):,} rows, {measured:,} with a fitted pIC50)")
        skeleton[f"comp_chem/pubchem/cyp_inhibition/{name}.csv"] = {
            "num_compounds": int(df["sid"].nunique()),
            "columns": {c: "" for c in df.columns},
        }

    print(f"\nWrote {len(skeleton)} files -> {args.output_dir}")
    print("\ndescriptions.json skeleton (fill in column meanings, merge into descriptions.json):")
    print(json.dumps(skeleton, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
