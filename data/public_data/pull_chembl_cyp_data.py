"""Pull public CYP inhibition potency from ChEMBL 37 into public-data CSVs.

Dose-response pIC50 for five human CYP isoforms (1A2, 2C9, 2C19, 2D6, 3A4),
aggregated to one row per compound. Shaped to sit alongside the OpenADMET CYP
challenge training set (`comp_chem/openadmet/cyp/`), which covers four of the
same isoforms.

Two families are written, from the same query and the same aggregation:

`cyp_inhibition/` is fitted curves only. ChEMBL assigns a `pchembl_value` only
where a real concentration-response curve was fitted, so compounds too weak to
produce one are systematically absent: nothing here falls below pIC50 4.0, while
24% of the challenge set does -- 9% of its CYP2D6 labels, 40% of its CYP3A4 ones.
It is pretraining data, not extra rows for a
calibrated regressor. Read the potency-enrichment note in descriptions.json first.

`cyp_inhibition/censored/` is the same table plus the weak tail those files drop.
"IC50 > 20 uM" is a real observation that a compound is weaker than the highest
concentration tested, and it is the only public evidence ChEMBL carries for
thousands of compounds. Where an isoform has such a record and no fitted curve,
the target column holds that per-row bound and `{iso}_pic50_lt` marks the row.
Train on these with `bounded_loss=True` so a prediction is penalized only when it
crosses the bound; with bounded loss off the bounds read as exact labels and the
files are worse than the fitted-only ones. Every other column matches.

Reads the Scigantic parquet mirror of ChEMBL over DuckDB rather than downloading
the 30GB SQLite release. Only the four tables below are touched, ~280MB total.

The shared `upload_data.py` then publishes both families and merges the matching
`descriptions.json` entries.

Run:
    python pull_chembl_cyp_data.py
    AWS_PROFILE=scp_sandbox_admin python upload_data.py --apply   # then publish

On each run this prints a per-file row count, column list, and a paste-ready
`columns` skeleton to help keep `descriptions.json` in sync with the real schema.
"""

import argparse
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
from rdkit import Chem, RDLogger

from pull_common import standardize_smiles

log = logging.getLogger("workbench")
RDLogger.DisableLog("rdApp.*")

OUTPUT_DIR = Path(__file__).parent / "output" / "comp_chem" / "chembl" / "cyp_inhibition"
CHALLENGE_DIR = Path(__file__).parent / "output" / "comp_chem" / "openadmet" / "cyp"

RELEASE = "chembl_37"
MIRROR = f"s3://scigantic-chembl/{RELEASE}/parquet"

# Human targets, pinned by ChEMBL ID. `pref_name` alone is ambiguous -- ChEMBL
# carries mouse and rat orthologs named "Cytochrome P450 1A2" as well.
TARGETS = {
    "CHEMBL3356": "cyp1a2",
    "CHEMBL3397": "cyp2c9",
    "CHEMBL3622": "cyp2c19",
    "CHEMBL289": "cyp2d6",
    "CHEMBL340": "cyp3a4",
}
ISOFORMS = sorted(TARGETS.values())

# Correctness filters, not taste. `confidence_score` is ChEMBL's target-assignment
# confidence; >= 8 keeps direct single-protein assignments.
MIN_CONFIDENCE = 8

# The potency types ChEMBL summarizes into `pchembl_value`. The censored arm reuses
# the set so both describe the same kind of measurement rather than two different ones.
POTENCY_TYPES = ("AC50", "Potency", "IC50", "Ki", "Kd", "EC50")


def measurement_query(value_expr: str, extra_where: str) -> str:
    """A measurement query over the mirror; the arms differ only in value and filter."""
    return f"""
SELECT
    t.chembl_id           AS target_chembl_id,
    m.chembl_id           AS chembl_id,
    m.pref_name           AS pref_name,
    cs.canonical_smiles   AS canonical_smiles,
    {value_expr}          AS pic50,
    s.confidence_score    AS confidence_score
FROM read_parquet('{MIRROR}/activities.parquet') a
JOIN read_parquet('{MIRROR}/assays.parquet') s ON a.assay_id = s.assay_id
JOIN read_parquet('{MIRROR}/target_dictionary.parquet') t ON s.tid = t.tid
JOIN read_parquet('{MIRROR}/molecule_dictionary.parquet') m ON a.molregno = m.molregno
JOIN read_parquet('{MIRROR}/compound_structures.parquet') cs ON a.molregno = cs.molregno
WHERE t.chembl_id IN ({",".join(f"'{t}'" for t in TARGETS)})
  AND a.data_validity_comment IS NULL
  AND a.potential_duplicate = 0
  AND s.confidence_score >= {MIN_CONFIDENCE}
  AND {extra_where}
"""


EXACT_QUERY = measurement_query("a.pchembl_value", "a.pchembl_value IS NOT NULL AND a.standard_relation = '='")

# Left-censored: the assay saw no inhibition up to its highest concentration, so the
# true pIC50 sits at or below the bound this row carries. Each record brings its own.
CENSORED_QUERY = measurement_query(
    "9 - log10(a.standard_value)",
    f"""a.standard_relation IN ('>', '>=')
  AND a.standard_type IN ({",".join(f"'{t}'" for t in POTENCY_TYPES)})
  AND a.standard_units = 'nM'
  AND a.standard_value > 0""",
)


def fetch_measurements(query: str, label: str) -> pd.DataFrame:
    """One row per activity measurement, structures and isoform attached."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-east-1';")
    log.info(f"Querying the {RELEASE} mirror for {label} measurements (confidence >= {MIN_CONFIDENCE})")
    df = con.execute(query).df()
    df["isoform"] = df["target_chembl_id"].map(TARGETS)
    log.info(f"  {len(df):,} {label} measurements over {df['chembl_id'].nunique():,} ChEMBL compounds")
    return df


def inchi_keys(smiles: pd.Series) -> pd.Series:
    """InChIKey per SMILES, computed once per unique structure."""
    unique = {s: None for s in smiles.dropna().unique()}
    for s in unique:
        mol = Chem.MolFromSmiles(s)
        unique[s] = Chem.MolToInchiKey(mol) if mol else None
    return smiles.map(unique)


def challenge_keys(subdir: str) -> set[str]:
    """Standardized InChIKeys for every compound in a challenge split."""
    files = sorted((CHALLENGE_DIR / subdir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No challenge CSVs under {CHALLENGE_DIR / subdir}. Run pull_openadmet_data.py first -- "
            "the blinded test set is needed to guarantee this file is contamination-free."
        )
    smiles = pd.concat([pd.read_csv(f, usecols=["smiles"]) for f in files], ignore_index=True)["smiles"]
    keys = inchi_keys(smiles.map(standardize_smiles))
    log.info(f"  challenge {subdir}: {len(files)} files, {keys.nunique():,} unique structures")
    return set(keys.dropna())


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a standardized SMILES and InChIKey, dropping what RDKit cannot parse."""
    df = df.copy()
    df["smiles"] = df["canonical_smiles"].map(standardize_smiles)
    dropped = int(df["smiles"].isna().sum())
    if dropped:
        log.warning(f"Dropping {dropped:,} measurements whose SMILES could not be standardized")
        df = df[df["smiles"].notna()]
    df["inchi_key"] = inchi_keys(df["smiles"])
    return df[df["inchi_key"].notna()]


def per_isoform(df: pd.DataFrame) -> pd.DataFrame:
    """Median, count and spread per (structure, isoform), isoforms pivoted to columns.

    Public potency data disagrees with itself more than one lab's does, so each
    isoform keeps a median, a count and a spread rather than a single number.
    """
    stats = df.groupby(["inchi_key", "isoform"])["pic50"].agg(pic50="median", n="size", sd="std").reset_index()
    wide = stats.pivot(index="inchi_key", columns="isoform")
    wide.columns = [f"{iso}_{stat}" for stat, iso in wide.columns]
    return wide.reindex(columns=[f"{iso}_{stat}" for iso in ISOFORMS for stat in ("pic50", "n", "sd")])


def identity(df: pd.DataFrame) -> pd.DataFrame:
    """One identity row per structure. Salts and tautomers collapse onto one structure,
    so a ChEMBL ID is picked per InChIKey rather than assumed unique."""
    return (
        df.sort_values(["inchi_key", "chembl_id"])
        .groupby("inchi_key")
        .agg(
            smiles=("smiles", "first"),
            orig_smiles=("canonical_smiles", "first"),
            chembl_id=("chembl_id", "first"),
            pref_name=("pref_name", "first"),
            max_confidence_score=("confidence_score", "max"),
        )
    )


def assemble(identity_df: pd.DataFrame, wide: pd.DataFrame, bounds: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per structure: identity plus a pic50 / n / sd triple per isoform.

    With `bounds` supplied, a censored record fills any isoform the compound has no
    fitted curve for. A fitted curve always wins, and where a bound is used `{iso}_pic50`
    holds the bound itself, `{iso}_pic50_lt` marks the row, and `{iso}_n` / `{iso}_sd`
    describe the bounds rather than fitted values.
    """
    out = identity_df.copy()
    wide = wide.reindex(out.index)
    bounds = None if bounds is None else bounds.reindex(out.index)
    for iso in ISOFORMS:
        pic50, n, sd = f"{iso}_pic50", f"{iso}_n", f"{iso}_sd"
        if bounds is None:
            out[pic50], out[n], out[sd] = wide[pic50], wide[n], wide[sd]
        else:
            bound = wide[pic50].isna() & bounds[pic50].notna()
            out[pic50] = wide[pic50].where(~bound, bounds[pic50])
            out[n] = wide[n].where(~bound, bounds[n])
            out[sd] = wide[sd].where(~bound, bounds[sd])
            out[f"{pic50}_lt"] = bound
        out[n] = out[n].fillna(0).astype(int)
    return out.reset_index()


def aggregate(exact: pd.DataFrame, censored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The fitted-only table and the censored superset, in that order.

    The fitted table is built from the fitted arm alone and never touches the censored
    one, so adding censoring cannot move a value in the files that already ship.
    """
    exact, censored = standardize(exact), standardize(censored)
    exact_wide, censored_wide = per_isoform(exact), per_isoform(censored)

    exact_identity, censored_identity = identity(exact), identity(censored)
    only_censored = censored_identity[~censored_identity.index.isin(exact_identity.index)]
    full_identity = pd.concat([exact_identity, only_censored]).sort_index()

    return assemble(exact_identity, exact_wide), assemble(full_identity, exact_wide, censored_wide)


def finalize(df: pd.DataFrame, blinded: set[str], training: set[str]) -> pd.DataFrame:
    """Drop blinded-test contamination, flag challenge overlap, and number the rows."""
    contaminated = df["inchi_key"].isin(blinded)
    log.info(f"  blinded test compounds present: {int(contaminated.sum())} (dropped)")
    df = df[~contaminated].copy()

    df["n_measurements"] = df[[f"{iso}_n" for iso in ISOFORMS]].sum(axis=1)
    df["in_challenge_train"] = df["inchi_key"].isin(training)
    log.info(f"  challenge training compounds also present: {int(df['in_challenge_train'].sum())} (kept, flagged)")

    identity = ["inchi_key", "smiles", "orig_smiles", "chembl_id", "pref_name", "n_measurements"]
    identity += ["max_confidence_score"]
    per_iso = [c for iso in ISOFORMS for c in (f"{iso}_pic50", f"{iso}_pic50_lt", f"{iso}_n", f"{iso}_sd")]
    ordered = identity + [c for c in per_iso if c in df.columns] + ["in_challenge_train"]

    df = df[ordered].sort_values("chembl_id").reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df


def split_by_isoform(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The wide table plus one file per isoform, filtered to rows carrying that label."""
    out = {"all_isoforms": df}
    for iso in ISOFORMS:
        keep = ["id", "inchi_key", "smiles", "orig_smiles", "chembl_id", "pref_name"]
        keep += [c for c in (f"{iso}_pic50", f"{iso}_pic50_lt", f"{iso}_n", f"{iso}_sd") if c in df.columns]
        keep += ["max_confidence_score", "in_challenge_train"]
        out[iso] = df.loc[df[f"{iso}_pic50"].notna(), keep].reset_index(drop=True)
    return out


def pull_cyp_data() -> dict[str, dict[str, pd.DataFrame]]:
    """Both families, keyed by output subdirectory ("" is the fitted-curve-only one)."""
    fitted_raw, censored_raw = aggregate(
        fetch_measurements(EXACT_QUERY, "fitted"), fetch_measurements(CENSORED_QUERY, "censored")
    )
    blinded, training = challenge_keys("testing"), challenge_keys("training")

    log.info("Fitted curves only:")
    fitted = finalize(fitted_raw, blinded, training)
    log.info("With censored bounds:")
    censored = finalize(censored_raw, blinded, training)
    for iso in ISOFORMS:
        log.info(f"  {iso}: {int(censored[f'{iso}_pic50_lt'].sum()):,} censored labels")

    # `id` is file-local and dense in each family; join across the two on `inchi_key`.
    return {"": split_by_isoform(fitted), "censored": split_by_isoform(censored)}


def main():
    parser = argparse.ArgumentParser(description="Pull CYP inhibition potency from the ChEMBL mirror")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    skeleton = {}
    print("\n" + "=" * 70)
    print(f"ChEMBL CYP Inhibition Pull ({RELEASE})")
    print("=" * 70)
    for subdir, family in pull_cyp_data().items():
        for name, df in family.items():
            out_path = args.output_dir / subdir / f"{name}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            log.info(f"  {subdir + '/' if subdir else ''}{name}.csv  ({len(df):,} compounds)")
            entry = {
                "num_compounds": len(df),
                "license": "CC-BY-SA-3.0",
                "columns": {c: "" for c in df.columns},
            }
            bounds = [c for c in df.columns if c.endswith("_pic50_lt")]
            if bounds:
                entry["censoring"] = {
                    c[: -len("_lt")]: {
                        "direction": "left",
                        "flag_column": c,
                        "n_censored": int(df[c].sum()),
                        "bound_source": "the concentration each record reports the IC50 exceeded",
                    }
                    for c in bounds
                }
            key = "/".join(p for p in ("comp_chem/chembl/cyp_inhibition", subdir, f"{name}.csv") if p)
            skeleton[key] = entry

    print(f"\nWrote {len(skeleton)} files -> {args.output_dir}")
    print("\ndescriptions.json skeleton (fill in column meanings, merge into descriptions.json):")
    print(json.dumps(skeleton, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
