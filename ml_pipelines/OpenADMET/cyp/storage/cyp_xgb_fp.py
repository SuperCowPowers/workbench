"""CYP XGBoost UQ models on Morgan count fingerprints — one per isoform.

An ensemble-diversity arm, not an accuracy play. The chemprop members all read the
molecule the same way, so the two-member pools (CYP1A2, CYP2C9, CYP3A4) average two
highly correlated models. Substructure counts are a different representation
entirely, which is what an average needs.

Features are the single packed `fingerprint` column, registered as compressed so the
4096 Morgan counts stay one column and the framework unpacks them at training time.
Rebuilding needs an `openadmet_cyp_fp` FeatureSet: run the challenge rows through
`smiles-to-fingerprints-v1`, then `set_compressed_features(["fingerprint"])`.

XGBoost is single-target, so each scored isoform gets its own model. Single-target also
means the public ChEMBL and qHTS columns are unreachable here: they are separate
targets, and pooling them into the scored column needs a cross-assay offset the
auxiliary-head models avoid by construction. So this arm trains on challenge rows only,
roughly 1,300-2,300 per isoform.

Outcome: negative. The decorrelation was real -- 0.55-0.67 against the chemprop members,
against the 0.69-0.90 they share -- but the arm is 0.11-0.20 Pearson behind them, and an
average pays more for that than it collects for the independence. Pool deltas ran +0.001
to -0.007 against resolutions of 0.018-0.056.
"""

from workbench.api import FeatureSet, ModelFramework, ModelType

FS_NAME = "openadmet_cyp_fp"
VARIANT = "fp"
BASE_TAGS = ["openadmet_cyp", "xgboost", "regression", "fingerprints"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()
print(f"{len(df):,} compounds, compressed features: {fs.get_compressed_features()}")

for iso in ISOFORMS:
    target = f"{iso}_pic50_direct_inhibition"
    name = f"cyp-{VARIANT}-reg-xgb-{iso.removeprefix('cyp')}"
    tags = BASE_TAGS + [VARIANT, iso]

    # Single-target frameworks cannot mask a NaN label, so rows without this isoform's
    # measurement leave the training view entirely.
    exclude_ids = list(df.loc[df[target].isna(), "molecule_name"])
    print(f"{iso}: {len(df) - len(exclude_ids):,} labeled rows")

    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        feature_list=["fingerprint"],
        target_column=target,
        description=f"CYP {iso.upper()} XGBoost UQ on Morgan count fingerprints",
        tags=tags,
        exclude_ids=exclude_ids,
    )
    model.set_owner("openadmet_cyp")

    end = model.to_endpoint(tags=tags)
    end.set_owner("openadmet_cyp")
    end.test_inference()
    end.cross_fold_inference()
