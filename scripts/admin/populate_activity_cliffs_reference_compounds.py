"""Populate per-cliff public reference datasets of validated activity cliffs.

An "activity cliff" here is a compound whose measured property differs
substantially from its near-neighbors in Morgan fingerprint space, where the
difference has been chemically rationalized (not chalked up to measurement
error). These cases are diagnostic for fingerprint-based UQ pipelines —
feature-space proximity can't see them, so any UQ method built on top of
Morgan fingerprints will rate them as "typical, well-supported" predictions
even when the model is badly wrong.

This script keeps each cliff as its own CSV under a shared prefix so the set
can grow without disturbing existing consumers and so each cliff stays
self-contained:

    s3://workbench-public-data/comp_chem/reference_compounds/activity_cliffs/
        ├─ logd_example_1.csv
        ├─ logd_example_2.csv
        └─ ...

Each per-cliff CSV holds the cliff compound + its unique fingerprint
neighbors with measured values pulled live from the source Model's
fingerprint-proximity index and FeatureSet. The shared metadata (assay,
source model, chemistry rationale) lives both in-row (for portability) and
in the top-level descriptions.json entry keyed by the filename.

Access (read-only, unsigned):
    from workbench.api import PublicData
    df = PublicData().get("comp_chem/reference_compounds/activity_cliffs/logd_example_1")

Usage:
    python scripts/admin/populate_activity_cliffs_reference_compounds.py --dry-run
    python scripts/admin/populate_activity_cliffs_reference_compounds.py
    # Single cliff:
    python scripts/admin/populate_activity_cliffs_reference_compounds.py --only logd_example_1
"""

import argparse
import logging

import pandas as pd

from workbench.api import FeatureSet, Model

from reference_compounds_common import run_populate

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

PREFIX_KEY = "comp_chem/reference_compounds/activity_cliffs"


# Each entry is one validated activity cliff. The script resolves SMILES,
# measured values, and similarities live by pulling the model's UQModel +
# FeatureSet — so adding a new cliff is just adding a config dict here.
#
# Required fields:
#   name          → filename stem (one CSV per cliff)
#   source_model  → Workbench model where the cliff was observed
#   compound_id   → the cliff compound's ID in the source FeatureSet
#   assay         → target column name in the FeatureSet (e.g. "logd")
#   n_neighbors   → how many unique fingerprint neighbors to capture
#   rationale     → chemical explanation of why this cliff is real (gating;
#                   if you can't explain it, the entry doesn't belong here)
CLIFFS = [
    {
        "name": "logd_example_1",
        "source_model": "open-admet-chemprop-logd",
        "compound_id": "E-0016943",
        "assay": "logd",
        "n_neighbors": 10,
        "rationale": (
            "Cliff compound is a 4-methoxy-piperidine; neighbors are 3-methoxy-pyrrolidine "
            "analogs of the same scaffold. The 3-position methoxy on the pyrrolidine sits "
            "closer (beta) to the basic amine, lowering its pKa via inductive withdrawal. "
            "At the assay pH (~7.4) the pyrrolidines have a substantially larger neutral "
            "fraction -> higher logD. Morgan r=2 fingerprints can't encode the pKa shift "
            "and rate the structures as ~0.86 similar, so the model and the UQ both see "
            "a 'typical' prediction. Measured logD of 1.0 for the piperidine is "
            "consistent with the chemistry, not a measurement error."
        ),
    },
]


COLUMN_ORDER = [
    "id",
    "compound_id",
    "role",
    "smiles",
    "measured_value",
    "similarity",
    "assay",
    "source_model",
]


def build_cliff_dataframe(cliff: dict) -> pd.DataFrame:
    """Pull the cliff compound + its unique fingerprint neighbors from the live model."""
    log.info(f"  Loading model '{cliff['source_model']}' ...")
    model = Model(cliff["source_model"])
    prox = model.fp_prox_model()  # Morgan r=2, 2048-bit count fingerprints

    # Over-request neighbors to absorb FeatureSet replicate rows (same molecule
    # appearing multiple times), then dedup to unique neighbor IDs. Replicate
    # count varies by dataset; 4x is a safe overshoot for current open_admet.
    raw_k = max(cliff["n_neighbors"] * 4, 20)
    log.info(f"  Pulling {raw_k} raw neighbors for '{cliff['compound_id']}' (will dedup) ...")
    nbrs = prox.neighbors(cliff["compound_id"], n_neighbors=raw_k, include_self=False)
    unique_nbrs = (
        nbrs.drop_duplicates(subset="neighbor_id", keep="first").head(cliff["n_neighbors"]).reset_index(drop=True)
    )
    log.info(f"  Got {len(unique_nbrs)} unique neighbor(s).")

    # Pull SMILES + measured values from the source FeatureSet in one query.
    log.info(f"  Pulling FeatureSet '{model.get_input()}' ...")
    fs = FeatureSet(model.get_input())
    fs_df = fs.pull_dataframe()
    id_col = fs.id_column
    target = cliff["assay"]

    wanted_ids = [cliff["compound_id"], *unique_nbrs["neighbor_id"].tolist()]
    subset = (
        fs_df[fs_df[id_col].isin(wanted_ids)][[id_col, "smiles", target]]
        .drop_duplicates(subset=[id_col])
        .set_index(id_col)
    )

    # Build the rows: cliff first (similarity=1.0), then neighbors in order.
    rows = []
    cliff_row = subset.loc[cliff["compound_id"]]
    rows.append(
        {
            "compound_id": cliff["compound_id"],
            "role": "cliff",
            "smiles": cliff_row["smiles"],
            "measured_value": float(cliff_row[target]),
            "similarity": 1.0,
            "assay": cliff["assay"],
            "source_model": cliff["source_model"],
        }
    )
    for _, nbr in unique_nbrs.iterrows():
        nbr_id = nbr["neighbor_id"]
        if nbr_id not in subset.index:
            log.warning(f"    Neighbor '{nbr_id}' not in FeatureSet — skipping.")
            continue
        nbr_row = subset.loc[nbr_id]
        rows.append(
            {
                "compound_id": nbr_id,
                "role": "neighbor",
                "smiles": nbr_row["smiles"],
                "measured_value": float(nbr_row[target]),
                "similarity": float(nbr["similarity"]),
                "assay": cliff["assay"],
                "source_model": cliff["source_model"],
            }
        )

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMN_ORDER]


def describe_cliff(cliff: dict, df: pd.DataFrame) -> dict:
    """Build the descriptions.json entry for this cliff's CSV."""
    return {
        "description": (
            f"Validated activity cliff observed in model '{cliff['source_model']}' "
            f"on assay '{cliff['assay']}'. Cliff compound is {cliff['compound_id']} "
            f"with {len(df) - 1} unique fingerprint neighbors. "
            f"Rationale: {cliff['rationale']}"
        ),
        "columns": {
            "id": "Integer row index",
            "compound_id": "Original compound ID from the source FeatureSet",
            "role": "'cliff' for the cliff compound, 'neighbor' for its fingerprint neighbors",
            "smiles": "Canonical SMILES from the source FeatureSet",
            "measured_value": "Experimentally measured value of the assay",
            "similarity": (
                "Tanimoto similarity to the cliff compound (1.0 for the cliff itself); "
                "Morgan r=2, 2048-bit count fingerprints"
            ),
            "assay": "Property/endpoint being measured (e.g. 'logd')",
            "source_model": "Workbench model in which the cliff was observed",
        },
        "num_compounds": int(len(df)),
        "compound_id": cliff["compound_id"],
        "assay": cliff["assay"],
        "source_model": cliff["source_model"],
        "rationale": cliff["rationale"],
    }


def main():
    parser = argparse.ArgumentParser(description="Populate the validated activity-cliffs reference datasets")
    parser.add_argument("--dry-run", action="store_true", help="Print the DataFrame(s) without uploading")
    parser.add_argument("--only", help="Process only the named cliff (matches `name` field).")
    args = parser.parse_args()

    cliffs = [c for c in CLIFFS if (args.only is None or c["name"] == args.only)]
    if not cliffs:
        log.error(f"No cliff named '{args.only}' (available: {[c['name'] for c in CLIFFS]})")
        return

    for cliff in cliffs:
        log.info(f"\n=== {cliff['name']} ===")
        df = build_cliff_dataframe(cliff)
        csv_key = f"{PREFIX_KEY}/{cliff['name']}.csv"
        description = describe_cliff(cliff, df)
        run_populate(df, csv_key, description, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
