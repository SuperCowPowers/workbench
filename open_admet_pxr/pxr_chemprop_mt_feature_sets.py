"""Producer: multi-task PXR FeatureSet (PXR pEC50 + public logP + logD).

Builds one SMILES-keyed multi-task FeatureSet that the multi-task Chemprop models
consume. PXR pEC50 is the primary task; public logP (52k rows) and logD (4.2k) are
auxiliary tasks that supervise the shared MPNN encoder — the bet is that anchoring
the representation to lipophilicity (PXR's dominant driver) transfers better to the
held-out analog series than a PXR-only model.

Chemprop builds its own graph features from SMILES, so no feature-endpoint pass is
needed — the FeatureSet is just [molecule_name, smiles, split, pec50, logp, logd],
with NaN where a source doesn't carry that target (chemprop masks missing targets).

Sources are unioned by canonical SMILES (a molecule in multiple sources collapses to
one row with all its known targets). The PXR `split` column is carried through as a
passthrough so the model can zero-weight the phase1_test rows out of training.

Run once before the multi-task phase model:  python pxr_chemprop_mt_feature_sets.py
"""

import pandas as pd
from workbench.api import DataSource, PublicData
from workbench.utils.multi_task import combine_multi_task_data, validate_multi_task_data

MT_FS = "openadmet_pxr_mt"
TARGETS = ["pec50", "logp", "logd"]  # primary first

# PXR primary task: train + revealed phase-1, with the split label preserved.
train = PublicData().get("comp_chem/openadmet_pxr/pxr_train")[["molecule_name", "smiles", "pec50"]].copy()
train["split"] = "train"
phase1 = (
    PublicData().get("comp_chem/openadmet_pxr/pxr_test_phase1_unblinded")[["molecule_name", "smiles", "pec50"]].copy()
)
phase1["split"] = "phase1_test"
pxr = pd.concat([train, phase1]).dropna(subset=["pec50"]).drop_duplicates("molecule_name").reset_index(drop=True)

# Auxiliary tasks: public logP / logD. Synthesize unique ids (these have no
# molecule_name); PXR is listed first, so on a SMILES collision its name wins.
logp = PublicData().get("comp_chem/logp/logp_all")[["smiles", "logp"]].dropna(subset=["logp"]).copy()
logp["molecule_name"] = [f"logp_{i}" for i in range(len(logp))]
logd = PublicData().get("comp_chem/logd/logd_all")[["smiles", "logd"]].dropna(subset=["logd"]).copy()
logd["molecule_name"] = [f"logd_{i}" for i in range(len(logd))]

df = combine_multi_task_data(
    dataframes=[pxr, logp, logd],
    target_columns=[["pec50"], ["logp"], ["logd"]],
    id_column="molecule_name",
    merge_on_smiles=True,  # public sources share no ids; collapse by canonical SMILES
    standardize_smiles=True,  # ChEMBL pipeline so cross-source SMILES actually match
    passthrough_columns=[["split"], [], []],  # carry PXR split through the merge
)
validate_multi_task_data(df, TARGETS, id_column="molecule_name")

DataSource(df, name=f"{MT_FS}_ds").to_features(
    MT_FS, id_column="molecule_name", tags=["openadmet_pxr", "multi_task", "activity"]
)
print(
    f"Built '{MT_FS}': {len(df)} rows — "
    f"pec50={df['pec50'].notna().sum()}, logp={df['logp'].notna().sum()}, logd={df['logd'].notna().sum()}"
)
