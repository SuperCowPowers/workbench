"""CYP spike: fine-tuned Atom-JEPA, one model per isoform, scored out of fold.

The fine-tuned counterpart of ml_pipelines/OpenADMET/cyp/cyp_tabicl_monroe.py: same
FeatureSets, one single-target model per isoform (multi-task is worth -0.002 mean on CYP,
per that script), unlabelled rows excluded. Trains through the spike template on a GPU
training job; model_framework=PYTORCH selects the pytorch_chem images and the GPU instance
ladder. No endpoint: scripts/cyp_compare.py scores the model's "training" capture, the fold
models' out-of-fold predictions.

`--readouts` trains on the readout FeatureSet, adding the screen model's four predicted
log2FC columns as features.

Each isoform is its own training job, so isoforms launched separately run in parallel:

    python cyp_atomjepa_finetune.py --isoforms cyp2d6
    python cyp_atomjepa_finetune.py --isoforms cyp3a4 --readouts
"""

import argparse
from pathlib import Path

from workbench.api import FeatureSet, ModelFramework, ModelType

TEMPLATE = Path(__file__).parent / "atomjepa.template"
FS_NAME = "openadmet_cyp_monroe_f1"
READOUT_FS = "openadmet_cyp_readout_monroe_f1"
ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
BASE_TAGS = ["openadmet_cyp", "atomjepa", "activity", "spike"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--readouts",
    action="store_true",
    help="Train on the readout FeatureSet, adding the four predicted log2FC columns as features",
)
parser.add_argument("--name-suffix", default=None, help="Append to the model name")
parser.add_argument("--isoforms", nargs="+", default=ISOFORMS, choices=ISOFORMS, help="Isoforms to build")
args = parser.parse_args()

fs = FeatureSet(READOUT_FS if args.readouts else FS_NAME)
df = fs.pull_dataframe()
readouts = [f"{iso}_log2fc_readout" for iso in ISOFORMS] if args.readouts else []

for iso in args.isoforms:
    target = f"{iso}_pic50_direct_inhibition"
    name = f"cyp-reg-atomjepa-{iso.removeprefix('cyp')}"
    if args.readouts:
        name += "-readout"
    if args.name_suffix:
        name += f"-{args.name_suffix.strip('-')}"

    exclude_ids = list(df.loc[df[target].isna(), "molecule_name"])
    print(f"{iso}: {len(df) - len(exclude_ids):,} labelled rows, excluding {len(exclude_ids):,} unlabelled")

    model = fs.to_model(
        name=name,
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.PYTORCH,
        custom_script=TEMPLATE,
        feature_list=["smiles"] + readouts,
        target_column=target,
        description=f"CYP {iso.upper()} pIC50, fine-tuned Atom-JEPA (spike)",
        tags=BASE_TAGS + [iso],
        exclude_ids=exclude_ids,
    )
    model.set_owner("open_admet_cyp")
