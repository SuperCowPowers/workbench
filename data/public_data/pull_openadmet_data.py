"""Pull the OpenADMET Consortium challenge datasets from HuggingFace.

Fetches every challenge CSV/TSV, snake_cases the column names (stripping units
in parentheses), and writes them under `output/comp_chem/openadmet/<challenge>/`.
The shared `upload_data.py` then publishes them to
`s3://workbench-public-data/comp_chem/openadmet/...` and merges the matching
entries from `descriptions.json`.

Challenges (all OpenADMET Consortium, openly licensed):
    pxr          PXR induction (Apache-2.0)
                 https://huggingface.co/datasets/openadmet/pxr-challenge-train-test
    expansionrx  9 ADMET endpoints (CC-BY-4.0)
                 https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data
    asap         ASAP/Polaris antiviral (MIT)
                 https://huggingface.co/datasets/openadmet/ASAP_Polaris_OpenADMET_challenge
    octant_cyp   Octant CYP inhibition / reactivity (Apache-2.0)
                 https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release

Run:
    python pull_openadmet_data.py                      # all challenges
    python pull_openadmet_data.py --challenge pxr asap
    AWS_PROFILE=scp_sandbox_admin python upload_data.py --apply   # then publish

On each run this prints a per-file row count, column list, and a paste-ready
`columns` skeleton to help keep `descriptions.json` in sync with the real schema.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("workbench")

OUTPUT_DIR = Path(__file__).parent / "output" / "comp_chem" / "openadmet"

# ExpansionRx: wide-file column -> per-endpoint output basename
RX_ENDPOINTS = {
    "logd": "logd",
    "ksol": "ksol",
    "hlm_clint": "hlm_clint",
    "mlm_clint": "mlm_clint",
    "caco_2_permeability_papp_a_b": "caco2_papp_a_b",
    "caco_2_permeability_efflux": "caco2_efflux",
    "mppb": "mppb",
    "mbpb": "mbpb",
    "mgmb": "mgmb",
}

# ASAP source headers that snake_casing alone would mangle (both pIC50 columns
# collapse to the same name once the parenthesised target is stripped).
ASAP_COLUMNS = {
    "ADMET.csv": {"MDR1-MDCKII": "mdr1_mdckii"},
    "Potency.csv": {"pIC50 (MERS-CoV Mpro)": "pic50_mers_mpro", "pIC50 (SARS-CoV-2 Mpro)": "pic50_sars_mpro"},
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + snake_case column names.

    Strips parenthesised units/annotations, collapses runs of underscores, and
    trims the ends, so a column like ``"pEC50 Std Error (-log10(molarity))"``
    becomes ``pec50_std_error``.
    """
    df = df.copy()
    cols = df.columns.str.lower()
    while cols.str.contains(r"\(").any():
        cols = cols.str.replace(r"\([^()]*\)", "", regex=True)
    cols = cols.str.replace(r"[^a-z0-9]+", "_", regex=True).str.replace(r"_+", "_", regex=True).str.strip("_")
    df.columns = cols
    return df


def fetch(repo: str, filename: str, sep: str = ",") -> pd.DataFrame:
    """Read one file out of a HuggingFace dataset repo."""
    url = f"hf://datasets/{repo}/{filename}"
    log.info(f"Fetching {url}")
    return pd.read_csv(url, sep=sep)


def pull_pxr() -> dict[str, pd.DataFrame]:
    """PXR induction challenge: train variants plus blinded / revealed test sets."""
    repo = "openadmet/pxr-challenge-train-test"
    files = {
        "pxr/training/main": "pxr-challenge_TRAIN.csv",
        "pxr/training/counter_assay": "pxr-challenge_counter-assay_TRAIN.csv",
        "pxr/training/single_concentration": "pxr-challenge_single_concentration_TRAIN.csv",
        "pxr/training/semi_pure_96": "pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv",
        "pxr/training/htchem_libraries": "pxr-challenge_htchem-libraries_TRAIN.csv",
        "pxr/testing/blinded": "pxr-challenge_TEST_BLINDED.csv",
        "pxr/testing/phase1_unblinded": "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "pxr/testing/structure_blinded": "pxr-challenge_structure_TEST_BLINDED.csv",
    }
    return {name: normalize_columns(fetch(repo, f)) for name, f in files.items()}


def pull_expansionrx() -> dict[str, pd.DataFrame]:
    """ExpansionRx: the wide 9-endpoint tables plus a long file per endpoint.

    Each per-endpoint file is the wide table restricted to compounds that have a
    measurement for that endpoint. Coverage varies a lot (MGMB 222 compounds,
    KSOL 5,128), so the wide table is mostly NaN and single-task work wants the
    per-endpoint files.
    """
    repo = "openadmet/openadmet-expansionrx-challenge-data"
    out = {}
    for split, filename in [("training", "expansion_data_train.csv"), ("testing", "expansion_data_test.csv")]:
        wide = normalize_columns(fetch(repo, filename)).rename(columns={"molecule_name": "id", **RX_ENDPOINTS})
        out[f"expansionrx/{split}/all_endpoints"] = wide
        for endpoint in RX_ENDPOINTS.values():
            single = wide[["id", "smiles", endpoint]].dropna(subset=[endpoint]).reset_index(drop=True)
            single.insert(2, "protocol", "open_admet")
            out[f"expansionrx/{split}/{endpoint}"] = single
    return out


def pull_asap() -> dict[str, pd.DataFrame]:
    """ASAP/Polaris antiviral challenge, split on the source `Set` column.

    Structures ship as CXSMILES; the enhanced-stereo annotation follows a space,
    so `smiles` is the plain prefix and `cxsmiles` keeps the full string.
    """
    repo = "openadmet/ASAP_Polaris_OpenADMET_challenge"
    out = {}
    for filename, basename in [("ADMET.csv", "admet"), ("Potency.csv", "potency")]:
        df = fetch(repo, filename).rename(columns=ASAP_COLUMNS[filename])
        df = normalize_columns(df).rename(columns={"molecule_name": "id"})
        df.insert(1, "smiles", df["cxsmiles"].str.split(" ").str[0])
        for split, label in [("training", "Train"), ("testing", "Test")]:
            rows = df[df["set"] == label].drop(columns=["set"]).reset_index(drop=True)
            out[f"asap/{split}/{basename}"] = rows
    return out


def pull_octant_cyp() -> dict[str, pd.DataFrame]:
    """Octant CYP inhibition / reactivity. A single release, no train-test split.

    `reactivity` ships keyed only on `ocnt_batch`; every batch in it also appears
    in `inhibition`, so the structure is joined in to make the file standalone.
    """
    repo = "openadmet/Octant_CYP_inhibition_reactivity_blog_release"
    inhibition = normalize_columns(fetch(repo, "inhibition.tsv", sep="\t")).rename(
        columns={"ocnt_batch": "id", "standardized_smiles": "smiles"}
    )
    inhibition.insert(1, "smiles", inhibition.pop("smiles"))
    reactivity = normalize_columns(fetch(repo, "reactivity.tsv", sep="\t")).rename(columns={"ocnt_batch": "id"})
    reactivity = reactivity.merge(inhibition[["id", "smiles"]], on="id", how="left")
    reactivity.insert(1, "smiles", reactivity.pop("smiles"))
    mass_spec = normalize_columns(fetch(repo, "will_it_fly_in_mass_spec.tsv", sep="\t")).rename(
        columns={"ocnt_batch": "id", "standardized_smiles": "smiles"}
    )
    return {
        "octant_cyp/inhibition": inhibition,
        "octant_cyp/reactivity": reactivity,
        "octant_cyp/mass_spec_response": mass_spec,
    }


CHALLENGES = {
    "pxr": pull_pxr,
    "expansionrx": pull_expansionrx,
    "asap": pull_asap,
    "octant_cyp": pull_octant_cyp,
}


def main():
    parser = argparse.ArgumentParser(description="Pull the OpenADMET challenge datasets from HuggingFace")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--challenge", nargs="*", choices=list(CHALLENGES), help="Only pull these challenges (default: all)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    skeleton = {}
    print("\n" + "=" * 70)
    print("OpenADMET Data Pull")
    print("=" * 70)
    for challenge in args.challenge or list(CHALLENGES):
        for name, df in CHALLENGES[challenge]().items():
            out_path = args.output_dir / f"{name}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            log.info(f"  {name}.csv  ({len(df):,} rows, {len(df.columns)} cols)  cols={df.columns.tolist()}")
            skeleton[f"comp_chem/openadmet/{name}.csv"] = {
                "num_compounds": int(len(df)),
                "columns": {c: "" for c in df.columns},
            }

    print(f"\nWrote {len(skeleton)} files -> {args.output_dir}")
    print("\ndescriptions.json skeleton (fill in column meanings, merge into descriptions.json):")
    print(json.dumps(skeleton, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
