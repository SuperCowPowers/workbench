# CYP challenge scripts

Utilities for the OpenADMET CYP challenge. These are not pipeline steps — the model
and FeatureSet builds live one directory up and are driven by `pipelines.json`.

| script | what it does |
|---|---|
| `cyp_submit.py` | Predicts the 750 blinded compounds and writes a validated Direct Inhibition entry |
| `cyp_recapture.py` | Rewrites the analog-holdout captures on the current code |

## openadmet_validation/

OpenADMET's own submission validators, vendored verbatim from the
[CYP-Challenge-Tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial)
(`validation/`), Apache-2.0, LICENSE included.

Vendored rather than reimplemented so the gate before uploading is the same code the
platform runs. Keep it verbatim — local edits would make it something other than their
validator, which defeats the point. Re-download from the tutorial repo if they revise
it mid-challenge.
