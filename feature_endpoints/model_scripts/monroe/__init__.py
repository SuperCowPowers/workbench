"""Monroe's encoder and featurizer, vendored for the `smiles-to-monroe-v1` feature endpoint.

Source: https://github.com/blazejba/monroe at commit 57238edfffea03808abe761a00cd9a75fa41bb95
(MIT; see LICENSE and NOTICE). The package layout matches upstream so its imports resolve.

Differences from upstream:
  - model/featurizer.py: `predict_structure` sets a fixed ETKDG seed and an embed timeout
    (the two lines marked `Workbench:`)
  - `__init__.py` and `model/__init__.py` import nothing (upstream's pull in training code)
"""
