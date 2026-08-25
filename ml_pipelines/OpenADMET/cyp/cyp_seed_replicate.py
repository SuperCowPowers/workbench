"""Replicate the current best config under a different fold seed.

Every candidate we compare from here sits on top of a noise floor, and this measures it.
`seed` shuffles the scaffold group ordering into folds; torch is never seeded, so a
replicate differs in both its fold assignment and its weight initialisation. Both of those
also differ between any two real candidates -- adding rows changes the scaffold groups --
so this is the floor a candidate delta has to clear to mean anything.

Config is identical to `cyp_chemprop_mt_aux_100.py` in every other respect, and that model
is the seed-42 replicate. Two more gives three points and a usable spread.

    python cyp_seed_replicate.py --seed 43
    python cyp_seed_replicate.py --seed 44

Read the spread with:  python scripts/cyp_seed_noise.py
"""

import argparse

import numpy as np
from workbench.api import FeatureSet, ModelFramework, ModelType
from workbench.utils.multi_task import compute_inverse_count_task_weights

FS_NAME = "openadmet_cyp_aux_f1"
TAGS = ["openadmet_cyp", "chemprop", "multi_task", "activity", "aux_log2fc", "seed_replicate"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
AUX_TARGETS = [f"{iso}_log2fc" for iso in ISOFORMS]
ALL_TARGETS = TARGETS + AUX_TARGETS

AUX_WEIGHT = 0.3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, required=True, help="Fold split seed; 42 is the standing model")
args = parser.parse_args()

model_name = f"cyp-reg-chemprop-mt-aux-100-s{args.seed}"

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()

primary_weights = compute_inverse_count_task_weights(df, TARGETS)
aux_weight = AUX_WEIGHT * float(np.mean(primary_weights))
task_weights = list(primary_weights) + [aux_weight] * len(AUX_TARGETS)
print(f"Building {model_name} on all {len(df)} rows, fold seed {args.seed}")

model = fs.to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=ALL_TARGETS,
    description=f"Seed replicate of cyp-reg-chemprop-mt-aux-100 (fold seed {args.seed})",
    tags=TAGS,
    hyperparameters={"task_weights": task_weights, "uq_version": "v1", "seed": args.seed},
)
model.set_owner("openadmet_cyp")

end = model.to_endpoint(tags=TAGS)
end.set_owner("openadmet_cyp")
end.test_inference()
end.cross_fold_inference()
