"""Producer: challenge pIC50s plus the Monroe embedding.

The four scored targets and their credible intervals, carried alongside the 720-d Monroe
embedding from `smiles-to-monroe-v1` as the compressed feature `monroe` (the XGBoost,
PyTorch and TabICL templates expand it; Chemprop's does not, so a Monroe model is a new
ensemble member rather than an upgrade to the existing ones).

Only `monroe` is a feature. Everything else is label metadata: the test set is SMILES and
a name, so a feature has to be derivable from structure, and the intervals are what makes
ST-RAE computable. Readouts -- predictions from a model trained on a larger related assay
-- are the other structure-derivable candidate and are not in this set.

Molecules Monroe cannot embed are dropped.

Run after the aux FeatureSet:  ml_pipeline_launcher cyp_monroe_feature_sets
"""

from workbench.api import DataSource, Endpoint, FeatureSet
from workbench.api.inference_cache import InferenceCache

SOURCE_FS = "openadmet_cyp_aux_f1"
MONROE_FS = "openadmet_cyp_monroe_f1"
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
LABEL_META = [f"{t}{suffix}" for t in TARGETS for suffix in ("_ci_lower", "_ci_upper", "_std")]

df = FeatureSet(SOURCE_FS).pull_dataframe()[["molecule_name", "smiles"] + TARGETS + LABEL_META]

# SMILES-keyed cache (S3-persisted), so a rebuild only embeds molecules it hasn't seen.
# A redeployed endpoint can change its output, so the cache resets when the endpoint does.
cached = InferenceCache(Endpoint("smiles-to-monroe-v1"), auto_invalidate_cache=True)
feat_df = cached.inference(df)

embedded = feat_df["monroe"].notna()
feat_df = feat_df[embedded]

DataSource(feat_df, name=f"{MONROE_FS}_ds").to_features(
    MONROE_FS, id_column="molecule_name", tags=["openadmet_cyp", "monroe", "activity"]
)
FeatureSet(MONROE_FS).set_compressed_features(["monroe"])

labelled = {t.split("_")[0]: int(feat_df[t].notna().sum()) for t in TARGETS}
print(f"Built '{MONROE_FS}': {len(feat_df)} of {len(df)} rows embedded; labelled per isoform {labelled}")
