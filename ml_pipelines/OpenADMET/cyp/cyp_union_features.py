"""Producer: the challenge FeatureSet unioned with public ChEMBL CYP potency.

`openadmet_cyp_aux_f1` carries 4,905 compounds. ChEMBL adds ~24,900 more with pIC50 for
five isoforms -- the four scored ones plus CYP2C19 -- and only ~185 structures overlap, so
this is almost entirely new chemistry rather than new labels on the same molecules. A
column join cannot express that; the rows have to be unioned and every target left NaN
where it was not measured, which is the shape chemprop's multi-task loss already expects.

ChEMBL potency goes in as its own five targets rather than merged into the scored ones.
On shared compounds the two assays correlate 0.31-0.66 and ChEMBL reads ~0.5 log more
potent, so merging would need a cross-assay affine correction that separate heads make
unnecessary: the encoder learns from both, each head keeps its own scale.

What this does NOT add is low-end range. ChEMBL assigns a `pchembl_value` only where a
concentration-response curve was fitted, so nothing sits below pIC50 4.0 -- the same
double selection that makes the challenge labels hit-enriched. Public potency data buys
ranking on new chemistry, not spread. Spread is a placement problem and is handled after
inference.

The Veith qHTS panel contributes `max_response` instead of its pIC50, which is 97%
redundant with ChEMBL. Efficacy at the top concentration is recorded for *every* compound
regardless of outcome -- 85,535 measurements at 100% coverage, against 50% for the pIC50 --
so the ~42,000 rows that showed no inhibition carry signal here rather than being dropped.
It correlates -0.53 to -0.77 with the challenge pIC50 on shared compounds.

That is the log2fc pattern, which is the only modelling change that has worked for us:
take the raw continuous readout rather than a label derived from it, and the censoring
problem disappears -- nothing is being bounded, the model predicts what the instrument
measured. `max_response` is negative for inhibition and is clipped to [-150, 50]. The readout is
percent change from control, so complete inhibition is -100 and -150 leaves a noise
margin, while anything above +50 is a compound "activating" the enzyme -- fluorescence
interference, not efficacy. Percentiles are the wrong tool here: the positive tail differs
by two orders of magnitude across isoforms (CYP2D6's 99.5th is +35, CYP2C9's is +290), so
a percentile clip keeps CYP2C9's artifacts and cuts CYP2D6's real signal. The fixed window
touches at most 3% of any isoform.

Every target lives here; the model scripts choose which to train on via `target_column`.
A FeatureSet is a data asset and the experiment variable belongs in the model, so adding
a source does not mean rebuilding data to isolate it.

Run after cyp_aux_features.py:  python cyp_union_features.py
"""

import pandas as pd
from rdkit import Chem, RDLogger
from workbench.api import DataSource, FeatureSet, PublicData

RDLogger.DisableLog("rdApp.*")

SOURCE_FS = "openadmet_cyp_aux_f1"
FS_NAME = "openadmet_cyp_union_f1"
CHEMBL = "comp_chem/chembl/cyp_inhibition/all_isoforms"

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
AUX_TARGETS = [f"{iso}_log2fc" for iso in ISOFORMS]
CI_COLUMNS = [f"{t}_{b}" for t in TARGETS for b in ("ci_lower", "ci_upper")]

# CYP2C19 is not scored by the challenge, but it is a correlated fifth task the public
# panel supplies for free.
PUBLIC_ISOFORMS = ISOFORMS + ["cyp2c19"]
CHEMBL_TARGETS = [f"{iso}_pic50_chembl" for iso in PUBLIC_ISOFORMS]
VEITH_TARGETS = [f"{iso}_max_response" for iso in PUBLIC_ISOFORMS]
VEITH = "comp_chem/pubchem/cyp_inhibition/all_isoforms"
MAX_RESPONSE_CLIP = (-150.0, 50.0)


def skeletons(smiles: pd.Series) -> pd.Series:
    """InChIKey connectivity block — matches structures across salt and stereo variants."""
    keys = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        keys.append(Chem.MolToInchiKey(mol).split("-")[0] if mol else None)
    return pd.Series(keys, index=smiles.index)


challenge = FeatureSet(SOURCE_FS).pull_dataframe()
challenge = challenge[["molecule_name", "smiles"] + TARGETS + AUX_TARGETS + CI_COLUMNS].copy()
challenge["key"] = skeletons(challenge["smiles"])

chembl = PublicData().get(CHEMBL)
chembl = chembl[["chembl_id", "smiles", "inchi_key"] + [f"{i}_pic50" for i in PUBLIC_ISOFORMS]].copy()
chembl = chembl.rename(columns={f"{i}_pic50": f"{i}_pic50_chembl" for i in PUBLIC_ISOFORMS})
chembl["key"] = chembl["inchi_key"].str.split("-").str[0]
chembl = chembl.drop(columns=["inchi_key"]).dropna(subset=["key"]).drop_duplicates(subset=["key"])

# Shared structures become one row carrying both label sets, keyed by the challenge's own
# identifier so the submission path and the analog split are unaffected.
shared = chembl[chembl["key"].isin(set(challenge["key"]))]
out = challenge.merge(shared[["key"] + CHEMBL_TARGETS], on="key", how="left")
if len(out) != len(challenge):
    raise ValueError(f"join changed the row count: {len(challenge)} -> {len(out)}")

new = chembl[~chembl["key"].isin(set(challenge["key"]))].copy()
new["molecule_name"] = new["chembl_id"]
out = pd.concat([out, new.drop(columns=["chembl_id"])], ignore_index=True)
out = out.drop(columns=["key"])

# --- Veith efficacy: joins onto whatever is already here, appends what is not ---------

veith = PublicData().get(VEITH)
wide = veith.pivot_table(index="smiles", columns="isoform", values="max_response")
wide.columns = [f"{c}_max_response" for c in wide.columns]
missing = [c for c in VEITH_TARGETS if c not in wide.columns]
if missing:
    raise ValueError(f"{VEITH} did not yield {missing} — isoform names changed?")
wide = wide[VEITH_TARGETS].reset_index()
wide[VEITH_TARGETS] = wide[VEITH_TARGETS].clip(lower=MAX_RESPONSE_CLIP[0], upper=MAX_RESPONSE_CLIP[1])
wide["key"] = skeletons(wide["smiles"])
wide = wide.dropna(subset=["key"]).drop_duplicates(subset=["key"])

out["key"] = pd.concat([challenge["key"], new["key"]], ignore_index=True).values
joined = out.merge(wide[["key"] + VEITH_TARGETS], on="key", how="left")
if len(joined) != len(out):
    raise ValueError(f"veith join changed the row count: {len(out)} -> {len(joined)}")

veith_only = wide[~wide["key"].isin(set(out["key"]))].copy()
veith_only["molecule_name"] = "VEITH-" + veith_only["key"]
out = pd.concat([joined, veith_only.drop(columns=["key"])], ignore_index=True)
out = out.drop(columns=["key"])

ALL_TARGETS = TARGETS + AUX_TARGETS + CHEMBL_TARGETS + VEITH_TARGETS
if out["molecule_name"].duplicated().any():
    raise ValueError("duplicate molecule_name after the union")
if out["smiles"].isna().any():
    raise ValueError("null smiles after the union")

print(f"challenge {len(challenge):,} + chembl-only {len(new):,} + veith-only {len(veith_only):,} = {len(out):,} rows")
print(f"{'target':<30} {'labelled':>9} {'coverage':>9}")
for target in ALL_TARGETS:
    n = int(out[target].notna().sum())
    print(f"{target:<30} {n:>9,} {100 * n / len(out):>8.1f}%")
print(f"total labels: {int(out[ALL_TARGETS].notna().sum().sum()):,}")

DataSource(out, name=f"{FS_NAME}_ds").to_features(
    FS_NAME, id_column="molecule_name", tags=["openadmet_cyp", "multi_task", "activity", "public"]
)
print(f"Built '{FS_NAME}': {len(out):,} rows, {len(ALL_TARGETS)} targets")
