"""Producer: the CYP regression FeatureSet with single-concentration log2fc as extra targets.

`openadmet_cyp_f1` carries only the four fitted-curve pIC50 targets, which between them
cover 6,525 measurements across 4,905 compounds -- 26-48% coverage per isoform. The
single-concentration arm measured 4,375 of those same compounds against all four enzymes
at 50 uM, adding 17,500 measurements at 89% coverage and spanning the inactive range the
dose-response track never reaches.

Those go in as four additional regression targets rather than as censored pIC50 bounds.
The bounded-loss attempt (`cyp_chemprop_mt_censored.py`) failed because CYP2D6's log2fc is
flat below pIC50 4.6, so it cannot say how far below a bound a compound sits. As a target
in its own right that flatness stops mattering -- the encoder learns what the assay
actually measured instead of a threshold we imposed on it.

log2fc is negative for inhibition and its scale differs per isoform (sd 0.46 for CYP2C9,
1.36 for CYP3A4), which chemprop handles by normalising each task.

Only the four pIC50 heads are ever submitted. The log2fc heads exist to supervise the
shared encoder.

Run after cyp_feature_sets.py:  python cyp_aux_features.py
"""

from workbench.api import DataSource, FeatureSet, PublicData

SOURCE_FS = "openadmet_cyp_f1"
FS_NAME = "openadmet_cyp_aux_f1"

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
AUX_TARGETS = [f"{iso}_log2fc" for iso in ISOFORMS]

df = FeatureSet(SOURCE_FS).pull_dataframe()
single_conc = PublicData().get("comp_chem/openadmet/cyp/training/single_concentration")

# One row per compound, one column per enzyme.
wide = single_conc.pivot_table(index="molecule_name", columns="enzyme", values="log2fc_estimate")
wide.columns = [f"{c.lower()}_log2fc" for c in wide.columns]
missing = [c for c in AUX_TARGETS if c not in wide.columns]
if missing:
    raise ValueError(f"single_concentration did not yield {missing} — enzyme names changed?")

out = df.merge(wide[AUX_TARGETS], left_on="molecule_name", right_index=True, how="left")
if len(out) != len(df):
    raise ValueError(f"join changed the row count: {len(df)} -> {len(out)}")

print(f"{'target':<24} {'labelled':>9} {'coverage':>9}")
for target in [f"{i}_pic50_direct_inhibition" for i in ISOFORMS] + AUX_TARGETS:
    n = int(out[target].notna().sum())
    print(f"{target:<24} {n:>9,} {100 * n / len(out):>8.1f}%")

DataSource(out, name=f"{FS_NAME}_ds").to_features(
    FS_NAME, id_column="molecule_name", tags=["openadmet_cyp", "multi_task", "activity", "aux_log2fc"]
)
print(f"Built '{FS_NAME}': {len(out):,} rows, {len(AUX_TARGETS)} auxiliary targets added")
