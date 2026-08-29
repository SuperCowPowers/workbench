# Retired CYP scripts

Kept for the record, not run by the pipeline. Each script's docstring carries its own
measurements; this is the index.

| script | what it built | outcome |
|---|---|---|
| `cyp_chemprop_mt.py` | four multi-task rotations, one per primary isoform | tie with the symmetric model at 4x cost, CI [-0.035, +0.021] |
| `cyp_pytorch_3dv2.py` | four single-task PyTorch on 2D + 3D-v2 | beaten by XGBoost on the same features, every isoform |
| `cyp_chemprop_mt_all.py` | the analog-holdout baseline | holdout retired: built from top hits, so active-enriched where the blind half is not |
| `cyp_chemprop_mt_100.py` | the first submission model | superseded by the auxiliary-target model |
| `cyp_chemprop_mt_aux.py` | analog-holdout counterpart of the aux model | holdout retired; its gains were never confirmed against the board |
| `cyp_censored_features.py` | CYP2D6 left-censored FeatureSet | see below |
| `cyp_chemprop_mt_censored.py` | bounded-loss model on it | negative, and the mechanism is the useful part: bounded loss has no gradient below the bound, so 2,627 rows at one bound collapsed to a constant that propagated through the shared encoder |
| `cyp_xgb_fp.py` | four single-task XGBoost on Morgan count fingerprints | decorrelated as hoped (0.55-0.67 vs chemprop) but 0.11-0.20 Pearson too weak; pool delta +0.001 to -0.007, inside noise on all four |

The isoform-weighting question these partly address is now settled: a 40x contrast on
CYP2D6's task weight moved its ranking by 0.004 on an 8,415-row ruler. Weighting is not
the lever -- the shared encoder is.

The fingerprint arm sets the other bound. A candidate earns a pool slot on decorrelation,
but only within roughly a tenth of the pool's best: the CYP2D6 specialists sit 0.01-0.06
behind at similar correlation and earn theirs, the fingerprint models sit 0.11-0.20 behind
and do not.
