"""Producer: PXR primary-screen FeatureSet (single-concentration log2FC).

The single-concentration screen sits upstream of the EC50 set: ~10.9k compounds
(~2.5x the pEC50 rows) read at 8.25 uM and 33 uM. It covers ~58% of the pEC50
training compounds and none of the phase-1 test compounds. One row per compound
(keyed by `ocnt_id`; `molecule_name` is blank on most screen rows), one column per
concentration (mean over replicates). The positive control and the two sparse
concentrations (99 uM, 0.98 uM) are dropped.

Consumed by pxr_chemprop_lfc.py, whose predictions become the "readout" features.

Run before the readout model:  ml_pipeline_launcher pxr_lfc_feature_sets
"""

from workbench.api import DataSource, PublicData

LFC_FS = "openadmet_pxr_lfc"
CONCENTRATIONS = {8.251e-06: "lfc_8um", 3.3e-05: "lfc_33um"}

sc = PublicData().get("comp_chem/openadmet/pxr/training/single_concentration")
sc = sc[sc["concentration_m"].isin(CONCENTRATIONS) & (sc["compound_class"] == "Library")]
lfc = (
    sc.pivot_table(index=["ocnt_id", "smiles"], columns="concentration_m", values="log2_fc_estimate")
    .rename(columns=CONCENTRATIONS)
    .reset_index()
)
lfc.columns.name = None

DataSource(lfc, name=f"{LFC_FS}_ds").to_features(LFC_FS, id_column="ocnt_id", tags=["openadmet_pxr", "primary_screen"])
counts = ", ".join(f"{c}={lfc[c].notna().sum()}" for c in CONCENTRATIONS.values())
print(f"Built '{LFC_FS}': {len(lfc)} compounds — {counts}")
