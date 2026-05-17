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
from tqdm import tqdm

# Single source of truth — shared with workbench.utils.multi_task.combine_multi_task_data.
from workbench.utils.chem_utils.mol_standardize import standardize_smiles  # noqa: F401  (re-exported)

log = logging.getLogger("workbench")


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
    priority_source: str | None = None,
) -> pd.DataFrame:
    """Save per-source CSVs and return a deduplicated dataframe keyed on canonical SMILES.

    Per-source files: ``{output_dir}/{file_prefix}_{source}.csv``.
    Merged columns: id, smiles, <value_name>, <value_name>_std, <value_name>_count,
    sources, primary_source, <value_name>_values.

    Args:
        priority_source: If given, the merged ``<value_name>`` for a compound is
            taken from this source whenever it reports the compound. Otherwise
            the cross-source mean is used. ``primary_source`` records which path
            was used per row (the source name, or ``"consensus"``).
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

    def reduce(group: pd.DataFrame) -> pd.Series:
        srcs = group["source"].tolist()
        vals = group[value_name].tolist()
        if priority_source is not None and priority_source in srcs:
            value = group.loc[group["source"] == priority_source, value_name].mean()
            primary = priority_source
        else:
            value = group[value_name].mean()
            primary = "consensus"
        return pd.Series(
            {
                value_name: value,
                std_col: group[value_name].std(),
                count_col: len(group),
                "sources": "|".join(sorted(set(srcs))),
                "primary_source": primary,
                values_col: "|".join(f"{v:.3f}" for v in vals),
            }
        )

    dedup = combined.groupby("canon_smiles").apply(reduce, include_groups=False).reset_index()
    dedup[std_col] = dedup[std_col].fillna(0.0)
    dedup[count_col] = dedup[count_col].astype(int)
    dedup = dedup.rename(columns={"canon_smiles": "smiles"})
    dedup.insert(0, "id", range(len(dedup)))

    # Only expose primary_source when a priority was actually applied — keeps
    # the legacy schema for single-priority pipelines (e.g., LogD) unchanged.
    cols = ["id", "smiles", value_name, std_col, count_col, "sources"]
    if priority_source is not None:
        cols.append("primary_source")
    cols.append(values_col)
    dedup = dedup[cols]

    log.info(f"Unique compounds after dedup: {len(dedup)}")
    if priority_source is not None:
        n_pri = (dedup["primary_source"] == priority_source).sum()
        log.info(f"  primary_source='{priority_source}': {n_pri:,} rows ({n_pri/len(dedup)*100:.1f}%)")
        log.info(f"  primary_source='consensus':       {len(dedup)-n_pri:,} rows")
    return dedup
