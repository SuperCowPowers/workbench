# OpenADMET PXR Challenge

Modeling work for the [OpenADMET PXR Induction Blind Challenge](https://openadmet.org/blindchallenges/)
(predicting human PXR induction, pEC50).

## Pipelines

`pipelines/` holds the model DAG as separate, self-contained scripts plus a
`pipelines.json` manifest (the FeatureSet stage is the producer; the model
stages consume it). Each script is mode-free and standalone, so the DAG can run
locally or be launched on AWS Batch with `ml_pipeline_launcher`, which discovers
the `pipelines.json`, orders the stages by their dependencies, and submits them.

- `pipelines/pxr_feature_sets.py` — DataSource + `_2d` / `_3d` / `_2d_3d` FeatureSets
- `pipelines/pxr_2d.py`, `pxr_3d.py`, `pxr_2d_3d.py` — XGBoost UQ + PyTorch UQ per feature block
- `pipelines/pxr_chemprop.py` — Chemprop D-MPNN (SMILES only)
- `pipelines/pipelines.json` — DAG manifest (fs producer → model consumers)

Every model endpoint captures `test` / `full` / `cross_fold` inference plus a
`pxr_phase1_test` capture on the held-out Analog Set 1
(`pxr_test_phase1_unblinded`, revealed pEC50) for honest external RAE.

Run order (locally):

```bash
cd pipelines
python pxr_feature_sets.py        # build FeatureSets first
python pxr_2d.py                  # then any/all model stages
python pxr_3d.py
python pxr_2d_3d.py
python pxr_chemprop.py
```

Or launch the whole DAG with the pipeline launcher (reads `pipelines.json` and
runs the stages in dependency order):

```bash
cd pipelines
ml_pipeline_launcher --dry-run --all   # preview the DAG order
ml_pipeline_launcher --local --all     # run the DAG locally
ml_pipeline_launcher --all             # submit the DAG to AWS Batch
```

- `pytorch_experiments.py` — standalone Chemprop / PyTorch experiments

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
