"""PXR primary-screen readout model: multi-task Chemprop on log2FC at 8.25 and 33 uM.

Learns the single-concentration screen (openadmet_pxr_lfc) so its predictions can
feed pEC50 models as "readout" features (pxr_readout_feature_sets.py). Chemprop masks
the NaN where a compound was read at only one concentration.

Build the FeatureSet first:  ml_pipeline_launcher pxr_lfc_feature_sets
"""

from workbench.api import FeatureSet, ModelType, ModelFramework

fs_name = "openadmet_pxr_lfc"
model_name = "pxr-reg-chemprop-lfc"
tags = ["openadmet_pxr", "chemprop", "primary_screen"]

m = FeatureSet(fs_name).to_model(
    name=model_name,
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    feature_list=["smiles"],
    target_column=["lfc_8um", "lfc_33um"],
    description="PXR primary-screen log2FC readout (multi-task: 8.25 uM + 33 uM)",
    tags=tags,
)
m.set_owner("open_admet_pxr")
end = m.to_endpoint(tags=tags)
end.set_owner("open_admet_pxr")
end.test_inference()
end.cross_fold_inference()
