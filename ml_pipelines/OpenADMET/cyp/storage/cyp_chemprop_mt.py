"""Multi-task Chemprop over the four scored CYP isoforms — one model per isoform.

All four isoforms are scored end products, so each gets its own model in which it is
the primary task and the other three are auxiliaries supervising the shared MPNN
encoder. Pooling all 6,525 dose-response measurements is chemprop's actual claim on
this challenge: the isoforms are correlated and the per-isoform data is small. It is
not that graph learning resolves activity cliffs — descriptor models match or beat
GNNs there — so the comparison worth running is this against single-task chemprop,
not against XGBoost.

`task_weights` keep the primary dominant in the gradient; the auxiliaries share a flat
0.3 since isoform coverage spans only ~1.8x (CYP3A4 2,335 curves, CYP2C9 1,285). The
framework scores a multi-task model on its first target, so each variant reports OOF
metrics, regression plots, and UQ calibration for its own isoform.

Evaluation is the analog holdout, not cross-validation: `analog_holdout_split`
reproduces how the challenge test set was built (top hits plus their nearest
neighbors), and a random split of the same size flatters a baseline by about 2x.
Every variant holds out the same rows via `validation_ids` and captures them
separately, so the four ST-RAE numbers are directly comparable.

`feature_list=["smiles"]` is explicit and load-bearing. The FeatureSet carries each
target's `_ci_lower`/`_ci_upper` so ST-RAE can be scored against them; an
auto-generated feature list would hand the model the bounds bracketing its own label.

Build the FeatureSet first: python cyp_feature_sets.py
"""

from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.training.splits import analog_holdout_split

FS_NAME = "openadmet_cyp_f1"
BASE_TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity"]

# CYP3A4 first (it metabolises roughly half of marketed drugs) — all four are scored.
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
CI_COLUMNS = [f"{t}_{bound}" for t in TARGETS for bound in ("ci_lower", "ci_upper")]
AUX_WEIGHT = 0.3

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

# Hold out potent hits and their close analogs across every target, mirroring the
# challenge's hit-expansion test set. Shared by all four variants.
holdout_mask = analog_holdout_split(df, target_columns=TARGETS, n_hits=50, analogs_per_hit=10)
holdout = df[holdout_mask]
validation_ids = list(holdout["molecule_name"])
holdout_df = holdout[["molecule_name", "smiles"] + TARGETS + CI_COLUMNS]
print(f"Analog holdout: {len(holdout)} of {len(df)} rows held out of training")

for iso in ISOFORMS:
    primary = f"{iso}_pic50_direct_inhibition"
    targets = [primary] + [t for t in TARGETS if t != primary]
    task_weights = [1.0] + [AUX_WEIGHT] * (len(targets) - 1)
    name = f"cyp-reg-chemprop-mt-{iso.removeprefix('cyp')}"
    tags = BASE_TAGS + [f"primary-{iso}"]

    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        feature_list=["smiles"],
        target_column=targets,  # scored on targets[0]; task_weights is positional
        description=f"Multi-task Chemprop, {iso.upper()} primary, analog holdout",
        tags=tags,
        hyperparameters={"task_weights": task_weights, "uq_version": "v1"},
        validation_ids=validation_ids,
    )
    model.set_owner("openadmet_cyp")

    end = model.to_endpoint(tags=tags)
    end.set_owner("openadmet_cyp")
    end.test_inference()
    end.cross_fold_inference()

    # Held-out capture on exactly the analog rows the model never trained on. The CI
    # columns ride along so ST-RAE is scored against the challenge's own credible intervals.
    end.inference(holdout_df, capture_name="cyp_analog_holdout")
