# OpenADMET PXR Challenge

Modeling work for the [OpenADMET PXR Induction Blind Challenge](https://openadmet.org/blindchallenges/)
(human PXR induction, pEC50). The write-up is
[A Weekend on the OpenADMET PXR Challenge](../../../docs/blogs/pxr_weekend_experiments.md).

## Held-out convention

The FeatureSets carry a `split` column marking each row `train` or `phase1_test` (Analog
Set 1, 253 compounds, a genuinely out-of-distribution chemotype). Every phase-1 model
holds the `phase1_test` rows out of training via `validation_ids` and then captures
predictions on exactly those rows as `pxr_phase1_test`, so held-out RAE is comparable
across every model here. Phase-2 models train on all rows and predict the blinded set.

## Layout

```
pipelines.json                        # one DAG; subdir-relative script paths
pxr_feature_sets.py                   # producer: openadmet_pxr_f1 (SMILES + 2D + 3D v1)
pxr_chemprop_mt_feature_sets.py       # producer: openadmet_pxr_mt (pEC50 + public logP/logD)
pxr_lfc_feature_sets.py               # producer: openadmet_pxr_lfc (single-concentration screen)
pxr_chemprop_lfc.py                   # the screen as a model, to be read out as features
pxr_readout_feature_sets.py           # producer: openadmet_pxr_readout (f1 + predicted log2FC)
pxr_readout_monroe_feature_sets.py    # producer: openadmet_pxr_readout_monroe (+ Monroe embedding)
phase1/
  pxr_chemprop_phase1.py              # the baseline every other model is measured against
  pxr_chemprop_hpo_phase1.py          # the same model with its knobs searched
  pxr_chemprop_mt_phase1.py           # multi-task: pEC50 + logP / logD / both
  pxr_chemprop_readout_phase1.py      # baseline + the screen readouts
  pxr_tabicl_monroe_readout_phase1.py # TabICL on Monroe embedding + readouts
  pxr_atomjepa_phase1.py              # Atom-JEPA, one model per pretrained encoder
phase2/
  pxr_chemprop_mt_phase2.py           # submission: multi-task trained on all rows
```

Three arms run off three producers: the plain `f1` FeatureSet feeds the baseline, its HPO
counterpart and Atom-JEPA; `openadmet_pxr_mt` feeds the multi-task models (the arm that beat
the baseline); and the readout chain feeds a screen prediction forward as a *feature*
(`lfc` FeatureSet → `lfc` model → readout FeatureSet → Chemprop and TabICL). A final
`workbench:models/model_promotion.py` node ranks the lot into `pxr-reg-v1`.

```bash
ml_pipeline_launcher --dry-run --all   # preview the DAG
ml_pipeline_launcher --all             # SQS -> Batch  (or --local --all)
ml_pipeline_launcher pxr_chemprop_phase1.py -- --split butina
```

## Data

The PXR datasets are published to the Workbench public data store:

```python
from workbench.api import PublicData
df = PublicData().get("comp_chem/openadmet/pxr/train/main")
```

Provisioning lives with the rest of the public-data tooling, not here:

- Pull: `data/public_data/pull_openadmet_data.py --challenge pxr`
- Publish: `data/public_data/upload_data.py --apply`

Source: <https://huggingface.co/datasets/openadmet/pxr-challenge-train-test> (Apache-2.0).
