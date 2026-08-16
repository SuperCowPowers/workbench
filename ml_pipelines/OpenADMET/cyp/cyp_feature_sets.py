"""Producer: CYP FeatureSets built from public data, ahead of the challenge release.

Two FeatureSets, each exercising a different part of the challenge setup so the
plumbing is proven before the real train/test files land:

  - openadmet_cyp_veith   multi-task, all four challenge isoforms plus CYP2C19.
                          Public Veith qHTS panel (PubChem AID 1851). This is the
                          shape the challenge model takes: sparse targets, NaN where
                          a compound has no fitted curve for that isoform.
  - openadmet_cyp_octant  single-task CYP3A4 with its credible-interval columns.
                          The only public source carrying CIs, so it is the only way
                          to exercise ST-RAE (see workbench.utils.metrics_utils)
                          end-to-end through a model and endpoint.

Chemprop builds its own graph features from SMILES, so neither needs a
feature-endpoint pass — the FeatureSets are SMILES + targets.

Veith is noisier than the challenge's own data (replicate compounds disagree by a
median of 0.40 pIC50 units, against Octant's 0.094 median interval width), so treat
it as pretraining signal rather than equal-weight training data.

Run once before the model scripts:  python cyp_feature_sets.py
"""

import pandas as pd
from workbench.api import DataSource, PublicData
from workbench.utils.multi_task import combine_multi_task_data, validate_multi_task_data

VEITH_FS = "openadmet_cyp_veith"
OCTANT_FS = "openadmet_cyp_octant"

# Challenge isoforms first (primary tasks), then CYP2C19 — not scored by the
# challenge, but a correlated fifth task the panel gives us for free.
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2", "cyp2c19"]
TARGETS = [f"{iso}_pic50" for iso in ISOFORMS]

# --- Multi-task FeatureSet over the Veith panel -------------------------------------

# One frame per isoform, each contributing its own pIC50 target. Rows without a fitted
# curve are dropped here rather than carried as NaN targets: the "Inactive" rows are
# censored (tested, no inhibition up to 57 uM) and modeling them needs a censored-loss
# decision we have not made yet.
frames = []
for isoform in ISOFORMS:
    df = PublicData().get(f"comp_chem/pubchem/cyp_inhibition/{isoform}")
    df = df[["sid", "smiles", "pic50"]].dropna(subset=["pic50"]).copy()
    df["id"] = df["sid"].astype(str)
    frames.append(df.rename(columns={"pic50": f"{isoform}_pic50"}).drop(columns=["sid"]))

veith = combine_multi_task_data(
    dataframes=frames,
    target_columns=[[t] for t in TARGETS],
    id_column="id",
    merge_on_smiles=True,  # same compound across isoform files, and replicate SIDs collapse
    standardize_smiles=False,  # the public pull already standardized `smiles`
)
validate_multi_task_data(veith, TARGETS, id_column="id")

DataSource(veith, name=f"{VEITH_FS}_ds").to_features(
    VEITH_FS, id_column="id", tags=["openadmet_cyp", "multi_task", "activity"]
)
coverage = ", ".join(f"{t}={veith[t].notna().sum()}" for t in TARGETS)
print(f"Built '{VEITH_FS}': {len(veith)} rows — {coverage}")

# --- Single-task Octant FeatureSet, carrying credible intervals ----------------------

# The CI columns ride along as ordinary columns so ST-RAE can be computed against them.
# compute_regression_metrics looks for `<target>_ci_lower` / `_ci_upper` by name.
octant = PublicData().get("comp_chem/openadmet/octant_cyp/inhibition")
octant = octant[
    ["id", "smiles", "cyp3a4_pic50", "cyp3a4_pic50_ci_lower", "cyp3a4_pic50_ci_upper", "drc_qc_status"]
].dropna(subset=["cyp3a4_pic50", "cyp3a4_pic50_ci_lower", "cyp3a4_pic50_ci_upper"])
octant = octant.drop_duplicates("id").reset_index(drop=True)

DataSource(octant, name=f"{OCTANT_FS}_ds").to_features(
    OCTANT_FS, id_column="id", tags=["openadmet_cyp", "activity", "credible_intervals"]
)
print(f"Built '{OCTANT_FS}': {len(octant)} rows with CYP3A4 pIC50 + credible intervals")
