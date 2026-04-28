"""Shared helpers for pull_logp_data.py and pull_logd_data.py.

Standardization, downloading, and merge/dedup logic that is identical between
the LogP and LogD pipelines. The two pull scripts only differ in their source
list and the value column name (logp vs logd).
"""

import io
import logging
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from tqdm import tqdm

from workbench.utils.chem_utils.mol_standardize import MolStandardizer

log = logging.getLogger("workbench")


# Full ChEMBL pipeline: cleanup → salt removal → charge neutralization → tautomer canonicalization.
# Multi-component mixtures with unknown large fragments are dropped.
_standardizer = MolStandardizer(canonicalize_tautomer=True, remove_salts=True, drop_mixtures=True)


def standardize_smiles(smiles: str) -> str | None:
    """Standardize a SMILES string. Returns canonical SMILES of the parent, or None if invalid."""
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


def standardize_df(
    df: pd.DataFrame,
    smiles_col: str,
    value_col: str,
    source: str,
    value_name: str,
) -> pd.DataFrame:
    """Standardize a dataframe to (smiles, canon_smiles, <value_name>, source)."""
    out = df[[smiles_col, value_col]].copy()
    out.columns = ["smiles", value_name]
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    out.dropna(subset=[value_name], inplace=True)
    out["canon_smiles"] = out["smiles"].apply(standardize_smiles)
    out.dropna(subset=["canon_smiles"], inplace=True)
    out["source"] = source
    return out[["smiles", "canon_smiles", value_name, "source"]].reset_index(drop=True)


def merge_and_deduplicate(
    frames: list[pd.DataFrame],
    output_dir: Path,
    value_name: str,
    file_prefix: str,
) -> pd.DataFrame:
    """Save per-source CSVs and return a deduplicated dataframe keyed on canonical SMILES.

    Per-source files: ``{output_dir}/{file_prefix}_{source}.csv``.
    Merged columns: id, smiles, <value_name>, <value_name>_std, <value_name>_count, sources, <value_name>_values.
    """
    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total rows before dedup: {len(combined)}")

    for source, grp in combined.groupby("source"):
        path = output_dir / f"{file_prefix}_{source}.csv"
        grp.to_csv(path, index=False)
        log.info(f"  Saved {len(grp):>6,} rows -> {path.name}")

    std_col = f"{value_name}_std"
    count_col = f"{value_name}_count"
    values_col = f"{value_name}_values"

    dedup = (
        combined.groupby("canon_smiles")
        .agg(
            mean_val=(value_name, "mean"),
            std_val=(value_name, "std"),
            count_val=(value_name, "count"),
            sources=("source", lambda x: "|".join(sorted(set(x)))),
            values=(value_name, lambda x: "|".join(f"{v:.3f}" for v in x)),
        )
        .reset_index()
    )
    dedup["std_val"] = dedup["std_val"].fillna(0.0)

    dedup = dedup.rename(
        columns={
            "canon_smiles": "smiles",
            "mean_val": value_name,
            "std_val": std_col,
            "count_val": count_col,
            "values": values_col,
        }
    )
    dedup.insert(0, "id", range(len(dedup)))
    dedup = dedup[["id", "smiles", value_name, std_col, count_col, "sources", values_col]]

    log.info(f"Unique compounds after dedup: {len(dedup)}")
    return dedup
