"""Multi-task Chemprop with one isoform deliberately up- or down-weighted.

A contrast, not a candidate. Our models differ on CYP2D6 by 0.01-0.03 Spearman while OOF
resolves 0.056 and the board 0.089, so nothing we have built can be told apart there --
and a ruler cannot be validated with models that do not differ. This makes them differ on
purpose.

    python cyp_chemprop_contrast.py --isoform cyp2d6 --primary-weight 4.0   # specialist
    python cyp_chemprop_contrast.py --isoform cyp2d6 --primary-weight 0.1   # starved

Two questions, one pair of builds:

  * whether CYP2D6 weighting does anything at all. The board's two best CYP2D6 entries are
    poor on the other three isoforms, which is what specialisation looks like, and the
    primary-weighted rotation we ran before came back "a tie" on a CI of [-0.035, +0.021]
    -- an interval containing every effect we care about. That was never a null result.
  * whether `cyp2d6_log2fc` works as a ruler. It has 4,375 rows against the target's 1,493
    and is weighted identically in every model we build, so it is uncontaminated -- but
    unvalidated. If the two arms differ, both rulers should see it and we learn whether
    they agree.

Everything else is held at the p30 configuration. Read with
`scripts/cyp_seed_noise.py --models ...` against the thresholds in
`scripts/cyp_ruler_power.py`.

Build the FeatureSet first: python cyp_union_features.py
"""

import argparse

import numpy as np
from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.utils.multi_task import compute_inverse_count_task_weights

FS_NAME = "openadmet_cyp_union_f1"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity", "public"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
PUBLIC_ISOFORMS = ISOFORMS + ["cyp2c19"]

# Scored targets first: the framework reports a multi-task model on its first target, and
# a bare capture's `prediction` column holds target[0].
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
LOG2FC_TARGETS = [f"{iso}_log2fc" for iso in ISOFORMS]
# The assay's other arms, same platform and chemistry as the scored targets.
TDI_TARGETS = [f"{iso}_pic50_tdi_condition" for iso in ISOFORMS]
EMAX_TARGETS = [f"{iso}_emax_vs_pos_ctrl_direct_inhibition" for iso in ISOFORMS]
ASSAY_TARGETS = LOG2FC_TARGETS + TDI_TARGETS + EMAX_TARGETS
PUBLIC_TARGETS = [f"{iso}_pic50_chembl" for iso in PUBLIC_ISOFORMS] + [f"{iso}_max_response" for iso in PUBLIC_ISOFORMS]
ALL_TARGETS = TARGETS + ASSAY_TARGETS + PUBLIC_TARGETS

# One weight for every challenge-assay auxiliary, at the value the log2fc arm was validated
# at, so adding heads does not silently re-tune the arm that already worked.
ASSAY_WEIGHT = 0.3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--isoform", required=True, choices=ISOFORMS, help="Isoform to re-weight")
parser.add_argument(
    "--primary-weight",
    type=float,
    required=True,
    help="Multiplier on that isoform's scored head; >1 specialises it, <1 starves it",
)
parser.add_argument(
    "--public-weight",
    type=float,
    default=0.30,
    help="Per-head weight for the ChEMBL and Veith targets, as a multiple of mean(primary)",
)
args = parser.parse_args()

tag = f"{args.isoform[3:]}-{str(args.primary_weight).replace('.', 'p')}"
model_name = f"cyp-reg-chemprop-contrast-{tag}"

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

primary_weights = list(compute_inverse_count_task_weights(df, TARGETS))
mean_primary = float(np.mean(primary_weights))
# Re-weight one scored head, leaving the other three and every auxiliary untouched, so the
# pair differs in exactly one number.
idx = TARGETS.index(f"{args.isoform}_pic50_direct_inhibition")
primary_weights[idx] *= args.primary_weight
assay_weight = ASSAY_WEIGHT * mean_primary
public_weight = args.public_weight * mean_primary
task_weights = primary_weights + [assay_weight] * len(ASSAY_TARGETS) + [public_weight] * len(PUBLIC_TARGETS)

aux_share = (assay_weight * len(ASSAY_TARGETS) + public_weight * len(PUBLIC_TARGETS)) / sum(task_weights)
print(f"Building {model_name} on all {len(df):,} rows — no holdout")
print(f"{args.isoform} scored head x{args.primary_weight}; all other heads unchanged")
print(f"pIC50 weights: {dict(zip(ISOFORMS, [round(float(w), 3) for w in primary_weights]))}")
print(f"assay weight:  {assay_weight:.3f} each ({ASSAY_WEIGHT} x mean primary), {len(ASSAY_TARGETS)} heads")
print(f"public weight: {public_weight:.3f} each ({args.public_weight} x mean primary), {len(PUBLIC_TARGETS)} heads")
print(f"auxiliary share of total gradient: {100 * aux_share:.0f}%")

model = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=ALL_TARGETS,
    description=f"Contrast build: {args.isoform} scored head x{args.primary_weight}",
    tags=TAGS + ["contrast", f"{args.isoform}_x{args.primary_weight}"],
    hyperparameters={"task_weights": task_weights, "uq_version": "v1"},
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()

# Out-of-fold over every row. The scored targets exist on 4,905 of them, which is the
# subset the arms are compared on.
end.cross_fold_inference()
