"""Example: tag molecules with curation tags for ADMET modeling.

Pipeline shown:
    1. Pull reference compounds from PublicData
    2. Standardize (salts/mixtures/tautomers via mol_standardize)
    3. Tag (namespaced categorical metadata via mol_tagging)
    4. Inspect: per-row tags, summary counts, ADMET training-set filter

The tagging reference set is purpose-built to light up every ``curation:*``
tag at least once — inorganics, organometallics, peptides, macrocycles,
isotopes, undefined stereo, halogen-heavy compounds, PAINS hits, MW
extremes — plus a handful of clean drug controls.
"""

import pandas as pd

from workbench.api import PublicData
from workbench.utils.chem_utils.mol_standardize import standardize
from workbench.utils.chem_utils.mol_tagging import (
    tag_molecules,
    get_tag_summary,
    admet_training_set,
    filter_by_tags,
)

pd.options.display.max_columns = None
pd.options.display.width = 1400
pd.options.display.max_colwidth = 100


# ---------------------------------------------------------------------------
# 1. Load reference compounds
# ---------------------------------------------------------------------------
REFERENCE_DATASET = "comp_chem/reference_compounds/tagging"
ref_df = PublicData().get(REFERENCE_DATASET)
print(f"Loaded {len(ref_df)} reference compounds from {REFERENCE_DATASET}\n")


# ---------------------------------------------------------------------------
# 2. Standardize (required before tagging — provides the canonical SMILES and
#    the 'undefined_chiral_centers' column used by curation:caution:stereo_undefined)
# ---------------------------------------------------------------------------
input_df = ref_df.rename(columns={"input_smiles": "smiles"})[["id", "name", "smiles"]].copy()
std_df = standardize(input_df)


# ---------------------------------------------------------------------------
# 3. Tag — all categories (composition, structure, physchem, liabilities, curation)
# ---------------------------------------------------------------------------
tagged = tag_molecules(std_df)


# ---------------------------------------------------------------------------
# 4. Inspect
# ---------------------------------------------------------------------------
print("\n--- Per-compound tags (first 15) ---")
print(tagged[["id", "name", "smiles", "tags"]].head(15))

print("\n--- Tag summary (top 25) ---")
print(get_tag_summary(tagged).head(25))

print("\n--- Compounds flagged for curation:exclude:* ---")
excluded = filter_by_tags(tagged, require_prefix=["curation:exclude:"])
print(excluded[["id", "name", "smiles", "tags"]])

print("\n--- Compounds flagged for curation:caution:* ---")
cautioned = filter_by_tags(tagged, require_prefix=["curation:caution:"])
print(cautioned[["id", "name", "tags"]].head(15))

print("\n--- ADMET training set (drops invalid + curation:exclude:*) ---")
train = admet_training_set(tagged)
print(f"Kept {len(train)}/{len(tagged)} compounds for general ADMET training.")

print("\n--- Strict training set (also drops curation:caution:*) ---")
strict = admet_training_set(tagged, drop_cautions=True)
print(f"Kept {len(strict)}/{len(tagged)} compounds for a strict small-molecule cut.")
