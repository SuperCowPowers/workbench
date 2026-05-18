"""Example: curate a public ADMET dataset for model training.

Shows the realistic workflow:
    1. Pull a real public ADMET dataset (LogP)
    2. Standardize (idempotent if already standardized; populates the
       undefined_chiral_centers column needed for the stereo caution tag)
    3. Tag every compound (composition, structure, physchem, liabilities,
       curation)
    4. Inspect the curation impact — what got dropped, why, and what stays
    5. Produce both a default training set and a strict training set

The companion script examples/chem_utils/tag_molecules.py walks through
what each individual tag means against a small reference set. This one
shows what the same tags do to a real ~50k-compound dataset.

Defaults to a 5,000-row random sample so the example runs in well under a
minute. Pass ``--full`` for the complete dataset (~10 min standardize +
tag on 52k compounds).
"""

import argparse

import pandas as pd

from workbench.api import PublicData
from workbench.utils.chem_utils.mol_standardize import standardize
from workbench.utils.chem_utils.mol_tagging import (
    tag_molecules,
    get_tag_summary,
    admet_training_set,
)

pd.options.display.max_columns = None
pd.options.display.width = 1400
pd.options.display.max_colwidth = 90


DATASET = "comp_chem/logp/logp_all"
DEFAULT_SAMPLE = 5000
RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Process the full {DATASET} dataset (~10 min). Default samples {DEFAULT_SAMPLE} rows.",
    )
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_SAMPLE, help=f"Sample size when not --full (default {DEFAULT_SAMPLE})"
    )
    args = parser.parse_args()

    # ---- 1. Load ----------------------------------------------------------
    print(f"Loading {DATASET}...")
    df = PublicData().get(DATASET)
    print(f"  full dataset: {len(df):,} compounds")

    if not args.full:
        df = df.sample(n=min(args.rows, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"  sampled:      {len(df):,} compounds (use --full for the complete dataset)\n")
    else:
        print()

    # ---- 2. Standardize ---------------------------------------------------
    # logp_all SMILES are already ChEMBL-standardized, but running standardize
    # is idempotent for already-canonical inputs and populates the
    # 'undefined_chiral_centers' column that the stereo caution tag reads.
    print("Standardizing...")
    df = standardize(df)
    invalid = df["smiles"].isna().sum()
    if invalid:
        print(f"  {invalid} rows failed standardization (dropped)")
        df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    print(f"  {len(df):,} standardized compounds\n")

    # ---- 3. Tag -----------------------------------------------------------
    print("Tagging...")
    tagged = tag_molecules(df)
    print()

    # ---- 4. Curation impact ----------------------------------------------
    print("=" * 70)
    print("Curation tag distribution (real LogP dataset)")
    print("=" * 70)
    summary = get_tag_summary(tagged)
    curation_summary = summary[summary.index.str.startswith("curation:")]
    if len(curation_summary):
        print(curation_summary.to_string())
    else:
        print("  (no curation flags fired — entire sample is clean)")
    print()

    # Liabilities (assay interference / structural alerts) — informational,
    # not auto-excluded by admet_training_set unless they propagate to a
    # curation:caution:pains tag.
    print("=" * 70)
    print("Liability tag distribution")
    print("=" * 70)
    liab_summary = summary[summary.index.str.startswith("liabilities:")]
    if len(liab_summary):
        print(liab_summary.to_string())
    print()

    # ---- 5. ADMET training set (default: drop excludes only) -------------
    print("=" * 70)
    print("Default ADMET training set (drops invalid + curation:exclude:*)")
    print("=" * 70)
    train = admet_training_set(tagged)
    pct = 100.0 * len(train) / len(tagged) if len(tagged) else 0
    print(f"Kept {len(train):,} / {len(tagged):,} compounds ({pct:.1f}%)\n")

    # Show a few example dropped rows per exclude reason for sanity-checking.
    print("Examples of dropped compounds, grouped by reason:")
    dropped = tagged[~tagged.index.isin(train.index)]
    exclude_tags = sorted(
        {t for tags in dropped["tags"] for t in tags if t.startswith("curation:exclude:")}
        | ({"invalid_smiles"} if (dropped["tags"].apply(lambda t: t == ["invalid_smiles"]).any()) else set())
    )
    for tag in exclude_tags:
        if tag == "invalid_smiles":
            sample = dropped[dropped["tags"].apply(lambda t: t == ["invalid_smiles"])]
        else:
            sample = dropped[dropped["tags"].apply(lambda t, _tag=tag: _tag in t)]
        print(f"\n  [{tag}] ({len(sample)} compounds)")
        print(sample[["smiles", "logp"]].head(3).to_string(index=False))
    print()

    # ---- 6. Strict ADMET training set ------------------------------------
    print("=" * 70)
    print("Strict ADMET training set (also drops curation:caution:*)")
    print("=" * 70)
    strict = admet_training_set(tagged, drop_cautions=True)
    strict_pct = 100.0 * len(strict) / len(tagged) if len(tagged) else 0
    print(f"Kept {len(strict):,} / {len(tagged):,} compounds ({strict_pct:.1f}%)")
    print(
        f"  ↳ Strict mode drops an additional {len(train) - len(strict):,} compounds "
        "vs. the default (peptides, macrocycles, undefined-stereo, etc.)"
    )
    print()

    # ---- 7. Sanity-check the kept set -----------------------------------
    print("=" * 70)
    print("Kept-set sanity check (default training set)")
    print("=" * 70)
    # MW range, LogP coverage
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    train_mw = train["smiles"].apply(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)))
    print(f"  MW:    min={train_mw.min():.0f}, median={train_mw.median():.0f}, max={train_mw.max():.0f}")
    print(
        f"  LogP:  min={train['logp'].min():.2f}, median={train['logp'].median():.2f}, "
        f"max={train['logp'].max():.2f}"
    )
    print(
        f"  Endpoint coverage: kept {train['logp'].notna().sum():,} non-null LogP "
        f"values ({100.0 * train['logp'].notna().sum() / len(train):.1f}% of kept set)"
    )


if __name__ == "__main__":
    main()
