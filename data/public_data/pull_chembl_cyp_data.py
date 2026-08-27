"""Pull public CYP inhibition potency from ChEMBL 37 into public-data CSVs.

Dose-response pIC50 for five human CYP isoforms (1A2, 2C9, 2C19, 2D6, 3A4),
aggregated to one row per compound. Shaped to sit alongside the OpenADMET CYP
challenge training set (`comp_chem/openadmet/cyp/`), which covers four of the
same isoforms.

Read the potency-enrichment note in descriptions.json before combining the two.
ChEMBL only assigns a `pchembl_value` where a real concentration-response curve
was fitted, so compounds too weak to produce one are systematically absent: this
file has nothing below pIC50 4.0, while ~40% of the challenge set is. It is
pretraining data, not extra rows for a calibrated regressor.

Reads the Scigantic parquet mirror of ChEMBL over DuckDB rather than downloading
the 30GB SQLite release. Only the four tables below are touched, ~280MB total.

Writes `output/comp_chem/chembl/cyp_inhibition/` -- one wide file over all
isoforms plus a per-isoform file filtered to measured rows. The shared
`upload_data.py` then publishes them and merges the matching
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

# Correctness filters, not taste. `confidence_score` is ChEMBL's target-assignment
# confidence; >= 8 keeps direct single-protein assignments.
MIN_CONFIDENCE = 8

QUERY = f"""
SELECT
    t.chembl_id           AS target_chembl_id,
    m.chembl_id           AS chembl_id,
    m.pref_name           AS pref_name,
    cs.canonical_smiles   AS canonical_smiles,
    a.pchembl_value       AS pic50,
    s.confidence_score    AS confidence_score
FROM read_parquet('{MIRROR}/activities.parquet') a
JOIN read_parquet('{MIRROR}/assays.parquet') s ON a.assay_id = s.assay_id
JOIN read_parquet('{MIRROR}/target_dictionary.parquet') t ON s.tid = t.tid
JOIN read_parquet('{MIRROR}/molecule_dictionary.parquet') m ON a.molregno = m.molregno
JOIN read_parquet('{MIRROR}/compound_structures.parquet') cs ON a.molregno = cs.molregno
WHERE t.chembl_id IN ({",".join(f"'{t}'" for t in TARGETS)})
  AND a.pchembl_value IS NOT NULL
  AND a.standard_relation = '='
  AND a.data_validity_comment IS NULL
  AND a.potential_duplicate = 0
  AND s.confidence_score >= {MIN_CONFIDENCE}
"""


def fetch_measurements() -> pd.DataFrame:
    """One row per activity measurement, structures and isoform attached."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-east-1';")
    log.info(f"Querying the {RELEASE} mirror ({len(TARGETS)} targets, confidence >= {MIN_CONFIDENCE})")
    df = con.execute(QUERY).df()
    df["isoform"] = df["target_chembl_id"].map(TARGETS)
    log.info(f"  {len(df):,} measurements over {df['chembl_id'].nunique():,} ChEMBL compounds")
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


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse measurements to one row per compound, isoforms pivoted to columns.

    Public potency data disagrees with itself more than one lab's does, so each
    isoform keeps a median, a count and a spread rather than a single number.
    """
    df = df.copy()
    df["smiles"] = df["canonical_smiles"].map(standardize_smiles)
    dropped = df["smiles"].isna().sum()
    if dropped:
        log.warning(f"Dropping {dropped:,} measurements whose SMILES could not be standardized")
        df = df[df["smiles"].notna()]
    df["inchi_key"] = inchi_keys(df["smiles"])
    df = df[df["inchi_key"].notna()]

    # Salts and tautomers collapse onto one structure, so a ChEMBL ID is picked
    # per InChIKey rather than assumed unique.
    per_iso = df.groupby(["inchi_key", "isoform"])["pic50"].agg(pic50="median", n="size", sd="std").reset_index()
    wide = per_iso.pivot(index="inchi_key", columns="isoform")
    wide.columns = [f"{iso}_{stat}" for stat, iso in wide.columns]
    wide = wide[[f"{iso}_{stat}" for iso in sorted(TARGETS.values()) for stat in ("pic50", "n", "sd")]]

    identity = (
        df.sort_values(["inchi_key", "chembl_id"])
        .groupby("inchi_key")
        .agg(
            smiles=("smiles", "first"),
            orig_smiles=("canonical_smiles", "first"),
            chembl_id=("chembl_id", "first"),
            pref_name=("pref_name", "first"),
            n_measurements=("pic50", "size"),
            max_confidence_score=("confidence_score", "max"),
        )
    )
    out = identity.join(wide).reset_index()
    for iso in TARGETS.values():
        out[f"{iso}_n"] = out[f"{iso}_n"].fillna(0).astype(int)
    return out


def pull_cyp_data() -> dict[str, pd.DataFrame]:
    """Fetch, aggregate, and split into a wide table plus one table per isoform."""
    df = aggregate(fetch_measurements())

    blinded = challenge_keys("testing")
    contaminated = df["inchi_key"].isin(blinded)
    log.info(f"Blinded test compounds present in ChEMBL: {contaminated.sum()} (dropped)")
    df = df[~contaminated]

    df["in_challenge_train"] = df["inchi_key"].isin(challenge_keys("training"))
    log.info(f"Challenge training compounds also in ChEMBL: {df['in_challenge_train'].sum()} (kept, flagged)")

    df = df.sort_values("chembl_id").reset_index(drop=True)
    df.insert(0, "id", range(len(df)))

    out = {"all_isoforms": df}
    for iso in sorted(TARGETS.values()):
        keep = ["id", "inchi_key", "smiles", "orig_smiles", "chembl_id", "pref_name"]
        keep += [f"{iso}_pic50", f"{iso}_n", f"{iso}_sd", "max_confidence_score", "in_challenge_train"]
        out[iso] = df.loc[df[f"{iso}_pic50"].notna(), keep].reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Pull CYP inhibition potency from the ChEMBL mirror")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    skeleton = {}
    print("\n" + "=" * 70)
    print(f"ChEMBL CYP Inhibition Pull ({RELEASE})")
    print("=" * 70)
    for name, df in pull_cyp_data().items():
        out_path = args.output_dir / f"{name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"  {name}.csv  ({len(df):,} compounds)")
        skeleton[f"comp_chem/chembl/cyp_inhibition/{name}.csv"] = {
            "num_compounds": len(df),
            "license": "CC-BY-SA-3.0",
            "columns": {c: "" for c in df.columns},
        }

    print(f"\nWrote {len(skeleton)} files -> {args.output_dir}")
    print("\ndescriptions.json skeleton (fill in column meanings, merge into descriptions.json):")
    print(json.dumps(skeleton, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
