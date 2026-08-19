# Retired CYP model scripts

Kept for the record, not run by the pipeline. Scored on the 529-compound analog
holdout, macro ST-RAE (lower is better, ~0.03 noise floor):

| script | models | macro ST-RAE |
|---|---|---|
| `cyp_chemprop_mt.py` | four rotations, one per primary isoform | 0.702 |
| `cyp_pytorch_3dv2.py` | four single-task PyTorch on 2D + 3D-v2 | 0.916 |

`cyp_chemprop_mt.py` builds one multi-task model per isoform, that isoform weighted
1.0 and the rest 0.3. Against the symmetric single model in `cyp_chemprop_mt_all.py`
(0.696) the paired bootstrap gives a delta of -0.006, 95% CI [-0.035, +0.021] — a
tie at four times the training cost.

`cyp_pytorch_3dv2.py` is beaten by the XGBoost models on the same features across
every isoform.
