# OpenADMET PXR Challenge

Modeling work for the [OpenADMET PXR Induction Blind Challenge](https://openadmet.org/blindchallenges/)
(predicting human PXR induction, pEC50). Scripts in this directory build the
FeatureSets and models:

- `create_feature_sets.py` — descriptor / fingerprint FeatureSets from the PXR data
- `all_models.py` — model training across feature/representation variants
- `pytorch_experiments.py` — Chemprop / PyTorch experiments

## Data

The PXR datasets are published to the Workbench public data store and accessed via:

```python
from workbench.api import PublicData
df = PublicData().get("comp_chem/openadmet_pxr/pxr_train")
```

Provisioning (fetch from HuggingFace → publish to S3) lives with the rest of the
public-data tooling, not here:

- Pull: `data/public_data/pull_pxr_data.py` (writes CSVs to `output/openadmet_pxr/`)
- Publish: `data/public_data/upload_data.py --apply` (uploads + merges `descriptions.json`)

Source: <https://huggingface.co/datasets/openadmet/pxr-challenge-train-test> (Apache-2.0).
