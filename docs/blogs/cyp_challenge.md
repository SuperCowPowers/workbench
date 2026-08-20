# OpenADMET CYP Challenge — Method Report

!!! tip inline end "Current entry"
    A deliberately plain baseline: one multi-task Chemprop model, stock hyperparameters, no ensembling and no tuning. It exists to establish an honest reference point, not to compete.

This is the standing method report for our entry to the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge) (Direct Inhibition track). It is updated as the entry changes; the section below always describes what is currently submitted.

## Current entry — a baseline (2026-08-20)

Our first submission is a **baseline**, built to calibrate internal evaluation against the leaderboard and to exercise the submission path end to end well before the November deadline.

Nothing in it is tuned. There is no ensemble, no hyperparameter search, no censored-data handling, and no TDI entry. Those are the next round; this one is the control they get measured against.

## The Model

A single multi-task Chemprop D-MPNN predicting all four isoforms from one shared encoder.

| | |
|---|---|
| **Architecture** | Chemprop D-MPNN, multi-task, one shared encoder over 4 targets |
| **Features** | SMILES only — the learned graph representation, no descriptors |
| **Targets** | CYP3A4, CYP2C9, CYP2D6, CYP1A2 direct-inhibition pIC50 |
| **Training data** | 4,905 compounds / 6,525 dose-response measurements (the challenge release, 100%) |
| **Task weights** | Inverse task counts, correcting uneven coverage (3A4 2,335 curves vs 2C9 1,285) |
| **Uncertainty** | Workbench `uq_version` v1 (not submitted; used internally to decide where to hedge) |
| **Hyperparameters** | Chemprop defaults |

**Why multi-task.** The isoforms are correlated and the per-isoform data is small — roughly 1,300–2,300 curves each. One encoder learning from all 6,525 measurements is a data-efficiency argument, not a claim that graph learning resolves activity cliffs; on cliffs, descriptor models match or beat GNNs. OpenADMET reach the same conclusion in their own tutorial, reporting that multi-task outperformed single-task in their internal experiments.

We also tested four *single-primary* variants (each isoform weighted 1.0 with the other three at 0.3). Against the symmetric single model they were a statistical tie — paired bootstrap delta of −0.006, 95% CI [−0.035, +0.021] — at four times the training cost, so the single model is what we submitted.

## Evaluation

The blind set is not a random draw. OpenADMET built it by hit expansion: the top 25 hits for CYP1A2, CYP2C9 and CYP3A4, then the top 10 catalog chemisimilars purchased for each — 3 × 25 × 10 = 750 compounds. It is dense clusters of near-neighbours around potent compounds.

So we evaluate on an **analog holdout** rather than cross-validation: take the top hits per target, hold out their nearest neighbours, and keep the hits themselves in training, which is how hit-expansion sets actually arise. On this data a random split of the same size flatters a baseline by roughly 2×.

The holdout lands on the real test set's difficulty. OpenADMET report the blind set's maximum ECFP4 Tanimoto to training centring around 0.50–0.55 with a tail to 0.7; our 529-compound holdout measures a median of 0.509 and a p90 of 0.646.

!!! note "Reading the numbers"
    **ST-RAE** is the challenge's metric: error is the distance from the prediction to the nearest bound of the label's Bayesian credible interval, and a prediction landing inside the interval scores zero. Lower is better. We use the plain `sum|y − ȳ|` denominator published in OpenADMET's tutorial. These are **our** holdout numbers, not leaderboard numbers.

| isoform | analog-holdout ST-RAE ↓ |
|---|---|
| CYP2C9 | 0.297 |
| CYP3A4 | 0.372 |
| CYP1A2 | 0.466 |
| CYP2D6 | 0.558 |
| **Macro** | **0.423** |

**Read that with suspicion, and we do.** Every compound in the training release was promoted to a dose-response curve *because* it was a primary-screen hit, so our holdout contains only pre-screened actives. The real 750 are unscreened catalog purchases and will contain many more inactives. We expect the leaderboard number to land meaningfully above 0.423, and how far above is the single most useful thing this submission tells us.

Model differences are quoted with a paired bootstrap (1,000 resamples) rather than as bare deltas. On this holdout the noise floor is about 0.03 macro ST-RAE, which is large enough to have swallowed at least one result we initially believed.

## Known Weakness: CYP2D6

The submitted predictions show CYP2D6 collapsing toward its mean:

| isoform | predicted σ | training σ | shrinkage | predicted below pIC50 4 | training below 4 |
|---|---|---|---|---|---|
| CYP3A4 | 0.90 | 1.09 | 0.82 | 21.6% | 40.4% |
| CYP2C9 | 0.72 | 0.78 | 0.92 | 13.5% | 20.2% |
| CYP1A2 | 0.65 | 1.03 | 0.63 | 9.5% | 16.4% |
| **CYP2D6** | **0.37** | **0.92** | **0.41** | **0.0%** | **8.6%** |

CYP2D6 predicts 41% of the real spread and calls **zero** of 750 compounds inactive. The mechanism is straightforward: it has the fewest sub-pIC50-4 training rows of any isoform, so the model has barely seen an inactive CYP2D6 compound and never predicts one. It is also the isoform with the narrowest credible intervals, so it forgives error least — and it is a quarter of the macro score.

We are reporting this rather than quietly hoping it doesn't bite. Fixing it is the first item of the next round.

## What This Baseline Deliberately Omits

- **Censored data.** Compounds showing no inhibition are real measurements (pIC50 below the assay floor), not missing values. The challenge's own single-concentration primary screen — 4,376 compounds against all four isoforms — contains exactly the inactives our training set lacks. Chemprop supports bounded loss; we have not used it here.
- **Ensembling.** Descriptor models are competitive on this task; a tabular model currently leads the leaderboard. Our XGBoost model on 2D + xTB 3D features trails Chemprop by 0.074 macro (99% on a paired bootstrap) but essentially ties on CYP2C9, making it a real ensemble member rather than a floor.
- **Hyperparameter search.** Prior work on the PXR challenge showed HPO gains on in-distribution cross-validation that reversed on the analog set. Stock defaults are the baseline to beat.
- **The TDI track.** No time-dependent-inhibition entry yet.

## Reproducing This

Built with [ADMET Workbench](https://github.com/SuperCowPowers/workbench). The full pipeline is four scripts:

```bash
# FeatureSets from the challenge release (2D + 3D features; Chemprop uses SMILES only)
python ml_pipelines/OpenADMET/cyp/cyp_feature_sets.py

# The evaluated model — holds out the analog set, reports honest ST-RAE
python ml_pipelines/OpenADMET/cyp/cyp_chemprop_mt_all.py

# The submitted model — same configuration, trained on 100% of the data
python ml_pipelines/OpenADMET/cyp/cyp_chemprop_mt_100.py

# Predict the 750 blinded compounds and write a validated submission
python ml_pipelines/OpenADMET/cyp/scripts/cyp_submit.py
```

The submission file is checked with OpenADMET's own validator, vendored from their
tutorial repository, so the gate before uploading is the same code the platform runs.

## References

- [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge)
- [CYP Challenge Tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial) — baseline notebooks, scoring harness, submission validators
- [A Weekend on the OpenADMET PXR Challenge](pxr_weekend_experiments.md) — our prior blind-challenge write-up, and where the HPO and 3D-descriptor priors come from
- van Tilborg et al., *Exposing the Limitations of Molecular Machine Learning with Activity Cliffs*, J. Chem. Inf. Model. 2022

## Questions?

Open an issue on [GitHub](https://github.com/SuperCowPowers/workbench) or reach us at [SuperCowPowers](https://www.supercowpowers.com).
