"""Pull mouse in-vivo PK reference compounds from ChEMBL 37 into a public-data CSV.

One row per compound carrying the four parameters that define a one-compartment
plasma curve: clearance, steady-state volume, oral bioavailability and plasma
free fraction. Only compounds with all four survive. The table is a reference
set for comparing predicted PK curves against measured ones; it is not training
data with a holdout guarantee.

Sources per parameter, all from assays ChEMBL annotates as mouse (tax id 10090):

    CL      in-vivo (BAO organism-based format), mL/min/kg
    Vdss    in-vivo, L/kg
    F       in-vivo, %, oral dosing (the assay description names the route;
            ChEMBL has no route field)
    fu      plasma or serum binding: Fu records (fraction) and PPB records
            (% bound, converted as 100 - PPB)

CL, Vdss and F come from one paper wherever a paper reports all three, so a row
describes one study's curve; ties go to the paper with the most measurements.
Compounds with no such paper fall back to the cross-paper median and carry
pk_same_paper = False. fu is the median over every qualifying record.

CL uses only the records from the best dosing route each structure has, in the
order iv, iv+po (one assay names both; CL conventionally comes from the IV arm),
unstated, po. cl_route names the route used. CL from oral dosing alone is likely
apparent clearance, CL/F, so po rows are kept but flagged.

Rows with a suspect value are dropped: fu of 0 (below quantitation) or >= 95%
(percent bound recorded as free), and CL outside 0.1-180 mL/min/kg (180 is twice
mouse hepatic blood flow; beyond it the records are unit errors).

Reads the Scigantic parquet mirror of ChEMBL over DuckDB, like
pull_chembl_cyp_data.py.

Run:
    python pull_chembl_mouse_pk_data.py
    AWS_PROFILE=scp_sandbox_admin python upload_data.py --apply   # then publish
"""

import argparse
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from pull_common import standardize_smiles

log = logging.getLogger("workbench")
RDLogger.DisableLog("rdApp.*")

OUTPUT_DIR = Path(__file__).parent / "output" / "comp_chem" / "chembl" / "mouse_pk"
DATASET = "mouse_pk_reference"

RELEASE = "chembl_37"
MIRROR = f"s3://scigantic-chembl/{RELEASE}/parquet"

MOUSE_TAX_ID = 10090
IN_VIVO = "BAO_0000218"  # BAO "organism-based format"
ORAL = r"oral|\bp\.?o\b|gavage"
IV = r"\bi\.?v\b|intravenous"

# CL dosing routes, best first (see module docstring)
ROUTE_PRIORITY = ["iv", "iv+po", "unstated", "po"]

# Suspect-value bounds (see module docstring)
CL_RANGE = (0.1, 180.0)
FU_MAX = 95.0

PK_PARAMS = ["cl_ml_min_kg", "vdss_l_kg", "f_percent"]
PARAMS = PK_PARAMS + ["fu_percent"]

QUERY = f"""
SELECT
    m.chembl_id           AS chembl_id,
    m.pref_name           AS pref_name,
    m.max_phase           AS max_phase,
    cs.canonical_smiles   AS canonical_smiles,
    d.doc_id              AS doc_id,
    d.chembl_id           AS document_chembl_id,
    a.standard_type       AS standard_type,
    a.standard_units      AS standard_units,
    a.standard_value      AS standard_value,
    s.bao_format          AS bao_format,
    s.assay_tissue        AS assay_tissue,
    lower(s.description)  AS description
FROM read_parquet('{MIRROR}/activities.parquet') a
JOIN read_parquet('{MIRROR}/assays.parquet') s ON a.assay_id = s.assay_id
JOIN read_parquet('{MIRROR}/docs.parquet') d ON a.doc_id = d.doc_id
JOIN read_parquet('{MIRROR}/molecule_dictionary.parquet') m ON a.molregno = m.molregno
JOIN read_parquet('{MIRROR}/compound_structures.parquet') cs ON a.molregno = cs.molregno
WHERE s.assay_tax_id = {MOUSE_TAX_ID}
  AND s.assay_type = 'A'
  AND a.standard_type IN ('CL', 'Vdss', 'F', 'Fu', 'PPB')
  AND a.standard_relation = '='
  AND a.standard_value IS NOT NULL
  AND a.data_validity_comment IS NULL
  AND a.potential_duplicate = 0
"""


def fetch_measurements() -> pd.DataFrame:
    """Every mouse ADME record of the five PK types, structures and documents attached."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-east-1';")
    log.info(f"Querying the {RELEASE} mirror for mouse CL/Vdss/F/Fu/PPB records")
    df = con.execute(QUERY).df()
    log.info(f"  {len(df):,} records over {df['chembl_id'].nunique():,} ChEMBL compounds")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per usable measurement: canonical parameter name, canonical unit."""
    t, u, v = df["standard_type"], df["standard_units"], df["standard_value"]
    in_vivo = df["bao_format"] == IN_VIVO
    plasma = df["assay_tissue"].isin(["Plasma", "Serum"])
    parts = [
        df[(t == "CL") & (u == "mL.min-1.kg-1") & in_vivo].assign(param="cl_ml_min_kg", value=v),
        df[(t == "Vdss") & (u == "L.kg-1") & in_vivo].assign(param="vdss_l_kg", value=v),
        df[(t == "F") & (u == "%") & in_vivo & v.between(0, 100) & df["description"].str.contains(ORAL)].assign(
            param="f_percent", value=v
        ),
        df[(t == "Fu") & u.isna() & plasma & v.between(0, 1)].assign(param="fu_percent", value=v * 100),
        df[
            (t == "PPB") & (u == "%") & plasma & v.between(0, 100) & ~df["description"].str.contains("unbound|free")
        ].assign(param="fu_percent", value=100 - v),
    ]
    out = pd.concat(parts, ignore_index=True)
    log.info(
        "  usable records per parameter: " + ", ".join(f"{p}={n:,}" for p, n in out["param"].value_counts().items())
    )
    return out


def inchi_keys(smiles: pd.Series) -> pd.Series:
    """InChIKey per SMILES, computed once per unique structure."""
    unique = {s: None for s in smiles.dropna().unique()}
    for s in unique:
        mol = Chem.MolFromSmiles(s)
        unique[s] = Chem.MolToInchiKey(mol) if mol else None
    return smiles.map(unique)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a standardized SMILES and InChIKey, dropping what RDKit cannot parse."""
    df = df.copy()
    df["smiles"] = df["canonical_smiles"].map(standardize_smiles)
    df["inchi_key"] = inchi_keys(df["smiles"])
    dropped = int(df["inchi_key"].isna().sum())
    if dropped:
        log.warning(f"Dropping {dropped:,} records whose SMILES could not be standardized")
    return df[df["inchi_key"].notna()]


def pk_values(df: pd.DataFrame) -> pd.DataFrame:
    """CL, Vdss and F per structure: one paper's values where a paper reports all three."""
    pk = best_route_cl(df[df["param"].isin(PK_PARAMS)])

    per_doc = summarize(pk, ["inchi_key", "doc_id"]).dropna(subset=PK_PARAMS)
    per_doc["total"] = per_doc[[f"{p}_n" for p in PK_PARAMS]].sum(axis=1)
    best = (
        per_doc.reset_index()
        .sort_values(["inchi_key", "total", "doc_id"], ascending=[True, False, True])
        .drop_duplicates("inchi_key")
        .set_index("inchi_key")
    )
    doc_ids = pk.drop_duplicates("doc_id").set_index("doc_id")["document_chembl_id"]
    same = best.drop(columns="total").assign(pk_same_paper=True, pk_documents=best["doc_id"].map(doc_ids))

    rest = pk[~pk["inchi_key"].isin(same.index)]
    cross = summarize(rest, ["inchi_key"]).assign(pk_same_paper=False, pk_documents=documents(rest))
    routes = pk[pk["param"] == "cl_ml_min_kg"].groupby("inchi_key")["cl_route"].first()
    return pd.concat([same.drop(columns="doc_id"), cross]).assign(cl_route=routes)


def best_route_cl(pk: pd.DataFrame) -> pd.DataFrame:
    """Keep each structure's CL records from its best dosing route only, labelled in `cl_route`."""
    iv = pk["description"].str.contains(IV)
    po = pk["description"].str.contains(ORAL)
    route = pd.Series(np.select([iv & po, iv, po], ["iv+po", "iv", "po"], "unstated"), index=pk.index)
    is_cl = pk["param"] == "cl_ml_min_kg"
    rank = route.map(ROUTE_PRIORITY.index).where(is_cl)
    best = rank.groupby(pk["inchi_key"]).transform("min")
    return pk.assign(cl_route=route)[~is_cl | (rank == best)]


def summarize(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Median and count per parameter, one flat column pair per parameter."""
    grouped = df.groupby(keys + ["param"])["value"]
    median = grouped.median().unstack("param")
    count = grouped.size().unstack("param").add_suffix("_n")
    return median.join(count)


def documents(df: pd.DataFrame) -> pd.Series:
    """Pipe-joined ChEMBL document IDs per structure."""
    return df.groupby("inchi_key")["document_chembl_id"].agg(lambda s: "|".join(sorted(set(s))))


def identity(df: pd.DataFrame) -> pd.DataFrame:
    """One identity row per structure; salts collapse, so a ChEMBL ID is picked per InChIKey."""
    return (
        df.sort_values(["inchi_key", "chembl_id"])
        .groupby("inchi_key")
        .agg(
            smiles=("smiles", "first"),
            orig_smiles=("canonical_smiles", "first"),
            chembl_id=("chembl_id", "first"),
            pref_name=("pref_name", "first"),
            max_phase=("max_phase", "max"),
        )
    )


def drop_suspect(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows carrying a value outside the plausible range."""
    rules = {
        "fu_percent == 0": df["fu_percent"] <= 0,
        f"fu_percent >= {FU_MAX}": df["fu_percent"] >= FU_MAX,
        f"cl_ml_min_kg outside {CL_RANGE}": ~df["cl_ml_min_kg"].between(*CL_RANGE),
    }
    for name, hit in rules.items():
        log.info(f"  suspect {name}: {int(hit.sum())}")
    suspect = pd.concat(rules, axis=1).any(axis=1)
    log.info(f"  dropped {int(suspect.sum())} suspect rows, kept {int((~suspect).sum())}")
    return df[~suspect]


def pull_mouse_pk() -> pd.DataFrame:
    """One row per structure with all four parameters and no suspect value."""
    df = standardize(normalize(fetch_measurements()))

    fu = df[df["param"] == "fu_percent"].groupby("inchi_key")["value"].agg(fu_percent="median", fu_percent_n="size")
    fu["fu_documents"] = documents(df[df["param"] == "fu_percent"])

    out = identity(df).join(pk_values(df), how="inner").join(fu, how="inner")
    out = out.dropna(subset=PARAMS)
    log.info(f"  structures with all four parameters: {len(out):,}")
    out = drop_suspect(out)

    for p in PARAMS:
        out[f"{p}_n"] = out[f"{p}_n"].astype(int)
    ordered = ["inchi_key", "smiles", "orig_smiles", "chembl_id", "pref_name", "max_phase"]
    ordered += PARAMS + [f"{p}_n" for p in PARAMS] + ["pk_same_paper", "cl_route", "pk_documents", "fu_documents"]
    out = out.reset_index()[ordered].sort_values("chembl_id").reset_index(drop=True)
    log.info("  cl_route: " + ", ".join(f"{route}={n}" for route, n in out["cl_route"].value_counts().items()))
    out.insert(0, "id", range(len(out)))
    return out


def main():
    parser = argparse.ArgumentParser(description="Pull mouse in-vivo PK reference compounds from the ChEMBL mirror")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("\n" + "=" * 70)
    print(f"ChEMBL Mouse PK Reference Pull ({RELEASE})")
    print("=" * 70)
    df = pull_mouse_pk()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{DATASET}.csv"
    df.to_csv(out_path, index=False)
    log.info(f"  {out_path.name}  ({len(df):,} compounds, {int(df['pk_same_paper'].sum()):,} single-paper)")

    skeleton = {
        f"comp_chem/chembl/mouse_pk/{DATASET}.csv": {
            "num_compounds": len(df),
            "license": "CC-BY-SA-3.0",
            "columns": {c: "" for c in df.columns},
        }
    }
    print("\ndescriptions.json skeleton (fill in column meanings, merge into descriptions.json):")
    print(json.dumps(skeleton, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
