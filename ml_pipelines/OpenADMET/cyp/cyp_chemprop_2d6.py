"""CYP2D6 with its own encoder — does dropping the sharing do what re-weighting could not?

Re-weighting CYP2D6's head by 40x moved its ranking by 0.004 on an 8,415-row ruler.
A head with enough capacity fits its target whatever the loss
says, so CYP2D6's ordering is coming from the shared encoder, which every other target also
shapes. That leaves one structural question the weighting experiment could not reach:
whether sharing an encoder across isoforms is what holds CYP2D6 back.

The board hints that it might. Its two best CYP2D6 entries are poor on the other three
isoforms -- the profile of a specialist -- while the entries strong everywhere sit mid-pack
on CYP2D6. CYP2D6 is also the isoform the challenge did not select hits on, whose
single-concentration readout is flat where the others are monotone, and the only one where
emax carries potency signal. Several independent reasons to think it is not like the other
three.

Two scopes, both with an encoder that sees CYP2D6 and nothing else:

    --scope single    the scored target alone, 1,493 rows. Maximum isolation, minimum data.
    --scope isoform   every CYP2D6 readout we have, ~16k rows. Drops cross-isoform sharing
                      while keeping within-isoform signal, which is the specialist design
                      an actual competitor would build.

Compare against `cyp-reg-chemprop-union-p30` on `cv_cyp2d6_pic50_direct_inhibition`, against
the thresholds in `scripts/cyp_ruler_power.py` -- 0.056 on the target itself, 0.031 on
log2fc. The training row set differs between variants, so scaffold folds differ too; that is
inherent to the comparison rather than a flaw, but it is not seed-for-seed.

If neither scope moves CYP2D6, representation sharing is not the problem and the remaining
hypothesis is features -- which is where the XGB-on-descriptors tie points.

Build the FeatureSet first: python cyp_union_features.py
"""

import argparse

from workbench.api import FeatureSet, ModelFramework, ModelType

FS_NAME = "openadmet_cyp_union_f1"
TAGS = ["openadmet_cyp", "chemprop", "activity", "cyp2d6_specialist"]

TARGET = "cyp2d6_pic50_direct_inhibition"
# Every other CYP2D6 readout: the challenge's own arms, then the public panels.
ISOFORM_AUX = [
    "cyp2d6_log2fc",
    "cyp2d6_pic50_tdi_condition",
    "cyp2d6_emax_vs_pos_ctrl_direct_inhibition",
    "cyp2d6_pic50_chembl",
    "cyp2d6_max_response",
]
AUX_WEIGHT = 0.3  # the value the auxiliary heads were validated at elsewhere

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scope",
    required=True,
    choices=["single", "isoform"],
    help="'single' trains on the scored target alone; 'isoform' adds every CYP2D6 readout",
)
args = parser.parse_args()

model_name = f"cyp-reg-chemprop-2d6-{args.scope}"
targets = [TARGET] if args.scope == "single" else [TARGET] + ISOFORM_AUX

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

trainable = int(df[targets].notna().any(axis=1).sum())
print(f"Building {model_name}: {len(targets)} target(s), {trainable:,} trainable rows of {len(df):,}")
for t in targets:
    print(f"  {t:<45}{int(df[t].notna().sum()):>7,}")

hyperparameters = {"uq_version": "v1"}
if len(targets) > 1:
    hyperparameters["task_weights"] = [1.0] + [AUX_WEIGHT] * len(ISOFORM_AUX)

model = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=targets,
    description=f"CYP2D6-only Chemprop, scope={args.scope}",
    tags=TAGS + [args.scope],
    hyperparameters=hyperparameters,
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()
end.cross_fold_inference()
