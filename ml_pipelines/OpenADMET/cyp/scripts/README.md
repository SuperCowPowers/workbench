# CYP challenge scripts

Utilities for the OpenADMET CYP challenge. These are not pipeline steps — the model
and FeatureSet builds live one directory up and are driven by `pipelines.json`.

| script | what it does |
|---|---|
| `cyp_submit.py` | Predicts the 750 blinded compounds from one model, writes a validated entry |
| `cyp_ensemble_submit.py` | Averages several models' blind-set predictions into one entry |
| `cyp_recalibrate.py` | Places a submission on the blind population; solves its moments from board rows |
| `cyp_mix_submission.py` | Assembles an entry from the per-isoform columns of several scored ones |
| `cyp_leaderboard.py` | Pulls the live boards from the HF Space |
| `cyp_compare.py` | Scores a model's captures, ST-RAE against the challenge's own intervals |
| `cyp_ruler_power.py` | What each ruler can resolve: paired row-sampling and seed noise |
| `cyp_seed_noise.py` | Reads seed replicates and reports the per-isoform noise floor |
| `cyp_member_diversity.py` | Whether a candidate arm earns a slot in an isoform's ensemble pool |
| `cyp_calibration_figures.py` | Writes the placement figure used in the blog |

## openadmet_validation/

OpenADMET's own submission validators, vendored verbatim from the
[CYP-Challenge-Tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial)
(`validation/`), Apache-2.0, LICENSE included.

Vendored rather than reimplemented so the gate before uploading is the same code the
platform runs. Keep it verbatim — local edits would make it something other than their
validator, which defeats the point. Re-download from the tutorial repo if they revise
it mid-challenge.
