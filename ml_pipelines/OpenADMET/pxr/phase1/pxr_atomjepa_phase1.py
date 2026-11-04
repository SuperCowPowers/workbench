"""PXR phase-1 Atom-JEPA: one model per pretrained encoder, on the shared FeatureSet (openadmet_pxr_f1).

`bio` is Atom-JEPA-Bio (the encoder after a second pretraining stage on PCBA bioassays);
`molecules` is the self-supervised encoder alone. Both are SMILES-only, like the other
contest members, and hold the `phase1_test` rows out of training via validation_ids, then
capture predictions on exactly those rows as 'pxr_phase1_test'.

Both encoders train before any endpoint is deployed, so a GPU capacity miss costs an
endpoint rather than the other model's training. Endpoints are async: GPU inference
images are refused as serverless, and async scales to zero when idle.

    ml_pipeline_launcher pxr_atomjepa_phase1.py
    ml_pipeline_launcher pxr_atomjepa_phase1.py -- --encoders bio
"""

import argparse

from workbench.api import FeatureSet, ModelFramework, ModelType

ENCODERS = ["bio", "molecules"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--encoders",
    nargs="+",
    default=ENCODERS,
    choices=ENCODERS,
    help="Encoders to build; defaults to both",
)
args = parser.parse_args()

fs = FeatureSet("openadmet_pxr_f1")
df = fs.pull_dataframe()
phase1 = df[df["split"] == "phase1_test"]

trained = []
for encoder in args.encoders:
    tags = ["openadmet_pxr", "atomjepa", encoder, "phase1"]
    model = fs.to_model(
        name=f"pxr-reg-atomjepa-{encoder}-phase1",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.ATOMJEPA,
        feature_list=["smiles"],
        target_column="pec50",
        description=f"PXR phase-1 pEC50 Atom-JEPA ({encoder} encoder; SMILES only; phase1_test held out of training)",
        tags=tags,
        validation_ids=list(phase1["molecule_name"]),  # held-out validation set (not trained)
        hyperparameters={"encoder": encoder},
    )
    model.set_owner("open_admet_pxr")
    trained.append((model, tags))

for model, tags in trained:
    end = model.to_endpoint(tags=tags, async_endpoint=True)
    end.set_owner("open_admet_pxr")
    end.test_inference()
    end.cross_fold_inference()

    # Held-out capture on the phase1_test rows (the model never trained on them)
    end.inference(phase1[["molecule_name", "smiles", "pec50"]], capture_name="pxr_phase1_test")
