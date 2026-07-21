# OpenADMET PXR Challenge

Modeling work for the [OpenADMET PXR Induction Blind Challenge](https://openadmet.org/blindchallenges/)
(predicting human PXR induction, pEC50).

## Layout

One **shared** FeatureSet (`openadmet_pxr_f1`) holds train + the revealed phase-1
set, with a `split` column marking each row. Both phase models (simple Chemprop,
SMILES-only) consume it; the phase-1 model holds the `phase1_test` rows out of
training via `validation_ids` so the held-out set never trains it. The xgb / pytorch / hybrid
explorations are parked in a top-level `storage/` (kept out of the phase dirs so
launching from inside a phase never picks them up).

```
pipelines.json         # one DAG: producer → phase1 + phase2 (subdir-relative paths)
pxr_feature_sets.py    # producer: builds the shared FeatureSet openadmet_pxr_f1
phase1/
  pxr_chemprop_phase1.py   # consumes the FS; holds out phase1_test; capture on it
phase2/
  pxr_chemprop_phase2.py   # consumes the FS; trains on all rows; predict 513 blinded → submission CSV
  activity_leaderboard_phase2.csv
storage/               # parked (*.py.archived): feature_sets, 2d/3d/2d_3d (xgb+pytorch), hybrid
```

- **phase1** — `pxr-reg-chemprop-phase1`. Runs `test` + `cross_fold` inference and a
  `pxr_phase1_test` capture on the held-out Analog Set 1 (`pxr_test_phase1_unblinded`,
  revealed pEC50) for honest external RAE.
- **phase2** — `pxr-reg-chemprop-phase2`. Trains on `pxr_train` + the now-revealed
  phase-1 set, predicts the blinded `pxr_test_blinded` (513 compounds), and writes
  `phase2_chemprop_submission.csv` (`SMILES, Molecule Name, pEC50`).

One launch from `open_admet_pxr/` does it all — `pipelines.json` declares the
dependency edges (producer → both models via `fs:openadmet_pxr_f1`) and the
launcher orders them. `storage/` is archived (`*.py.archived`) so the launcher's
`*.py` sweep skips it; rename a script back to `.py` to resurrect it.

```bash
cd open_admet_pxr
ml_pipeline_launcher --dry-run --all   # preview: feature_sets → phase1 + phase2
ml_pipeline_launcher --all             # SQS → Batch  (or --local --all)
```

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
