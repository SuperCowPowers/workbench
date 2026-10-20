"""ExpansionRx spike: fine-tuned Atom-JEPA on one Open ADMET assay.

Trains `{assay}-reg-atomjepa` on the assay's FeatureSet (built by
ml_pipelines/OpenADMET/load_data/load_data.py), SMILES only, 5 folds, with the Atom-JEPA
framework on a GPU training job. No endpoint: rx_atomjepa_scoreboard.py scores the out-of-fold
predictions (the "training" capture) against the assay's other models.

    python rx_atomjepa_finetune.py mppb
"""

import argparse

from workbench.api import FeatureSet, ModelFramework, ModelType

ASSAYS = ["logd", "ksol", "hlm_clint", "mlm_clint", "caco2_papp_a_b", "caco2_efflux", "mppb", "mbpb", "mgmb"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("assay", choices=ASSAYS)
args = parser.parse_args()

target = args.assay
fs = FeatureSet(f"open_admet_{target}")
df = fs.pull_dataframe()
exclude_ids = list(df.loc[df[target].isna(), "molecule_name"])
print(f"{target}: {len(df) - len(exclude_ids):,} labelled rows, excluding {len(exclude_ids):,} unlabelled")

model = fs.to_model(
    name=f"{target.replace('_', '-')}-reg-atomjepa",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.ATOMJEPA,
    feature_list=["smiles"],
    target_column=target,
    description=f"Fine-tuned Atom-JEPA for {target} prediction (spike)",
    tags=["open_admet", target, "regression", "atomjepa", "spike"],
    exclude_ids=exclude_ids,
)
model.set_owner("BW")
