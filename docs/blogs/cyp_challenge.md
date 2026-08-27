# OpenADMET CYP Challenge — Method Report

!!! tip inline end "Current entry"
    One multi-task Chemprop model with auxiliary targets, stock hyperparameters, no ensembling — plus a per-isoform placement correction. **Macro ST-RAE 0.4975, rank 2** as of 2026-08-27, and rank 1 on CYP1A2, CYP2C9 and CYP3A4 individually.

This is the standing method report for our entry to the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge) (Direct Inhibition track). It is updated as the entry changes; it always describes what is currently submitted.

One thing dominated everything else we tried: the training labels and the blind set are drawn from different populations, and correcting for that is worth more than any modelling change we made.

## The Model

A single multi-task Chemprop D-MPNN with eight heads on one shared encoder — the four scored pIC50 targets, plus four auxiliary targets that exist only to supervise the representation.

| | |
|---|---|
| **Architecture** | Chemprop D-MPNN, one shared encoder, 8 task heads |
| **Features** | SMILES only — the learned graph representation, no descriptors |
| **Scored targets** | pIC50 for CYP1A2, CYP2C9, CYP2D6, CYP3A4 |
| **Auxiliary targets** | Single-concentration log2 fold-change, one per isoform |
| **Training data** | 4,905 compounds / 6,525 dose-response curves + 17,500 single-concentration measurements (the challenge release, 100%) |
| **Task weights** | Inverse task counts on the scored targets; auxiliaries at 0.3× their mean |
| **Hyperparameters** | Chemprop defaults |

**Why multi-task.** The isoforms are correlated and the per-isoform data is small — roughly 1,300–2,300 curves each. One encoder learning from all 6,525 measurements is a data-efficiency argument, not a claim that graph learning resolves activity cliffs; on cliffs, descriptor models match or beat GNNs. OpenADMET report the same finding in their own tutorial.

**Why auxiliary targets.** The dose-response arm covers 26–48% of compounds per isoform. The single-concentration arm measured *every* compound against *every* isoform — 17,500 readings at 89% coverage — but its values are a fold-change, not a pIC50, so they cannot be pooled with the scored labels. As separate heads they do not need to be: they supervise the shared encoder and the pIC50 heads keep their own scale.

This is the single largest modelling gain we have found, and it landed exactly where we needed it:

| isoform | Pearson on a held-out analog set, without → with |
|---|---|
| CYP3A4 | 0.739 → 0.760 |
| CYP2C9 | 0.674 → 0.678 |
| **CYP2D6** | **0.419 → 0.502** |
| CYP1A2 | 0.588 → 0.619 |

CYP2D6 had resisted censored labels, architecture changes and extra data. Only its +0.083 clears the spread we measure between identical training runs at different random seeds (0.031 on CYP1A2 and CYP2D6, 0.008 and 0.007 on the other two); the remaining rows sit inside it and are not established.

It also sidesteps what killed our censored-label attempt below: the fold-change readout being flat at the low end stops mattering when it is a target to predict rather than a bound to respect.

## Two Populations

The blind set is not a random draw. OpenADMET built it by hit expansion: the top 25 hits for CYP1A2, CYP2C9 and CYP3A4, then the top 10 catalog chemisimilars purchased for each — 750 compounds in dense structural clusters around potent molecules.

Structurally clustered around hits is not the same as active. Analogs of a potent compound are mostly not potent, and CYP2D6 was not among the isoforms hits were selected on, so it received no enrichment at all. Meanwhile every training label survived two filters: a compound was screened because someone thought it might be active, and it was labelled only if its dose-response curve actually fit. Compounds that did nothing produce no curve and no label.

The two distributions therefore differ, and in a way you can estimate before submitting anything. The public PubChem qHTS panel puts CYP2D6 inactivity at roughly 65%; a set that is 65% inactive centres near pIC50 3.7, against the 4.69 our model predicted. The blind population is also *wider* than the training labels on all four isoforms — a squared-error model shrinks toward the mean, and a label set built from successful curve fits is already narrower than the population it came from.

## Placement

Predictions carry two independent things. Their **order**, which is the model, and their **placement** on the axis, which is not. R² decomposes exactly:

$$R^2 = 2\rho k - k^2 - b^2$$

with ρ the Pearson correlation, `k = sd(pred)/sd(true)` the spread ratio, and `b` the mean offset in units of sd(true). Only ρ depends on the ordering; `k` and `b` are set by an affine transform that touches no compound's rank. Two consequences: **R² ≤ ρ²** is a hard ceiling, and a model far below its own ρ² is *mis-placed, not weak*.

The optimum is **`k = ρ`, not `k = 1`**. Matching the spread of the truth is wrong — a model with ρ = 0.7 should be 70% as wide as reality, because shrinking toward the mean is the correct response to uncertainty. Ours were narrower even than that.

So: estimate the target population's centre and spread, then place each isoform there with spread `ρ·sd`.

<img src="../../images/cyp_calibration_applied.svg" alt="Four panels, one per isoform, each overlaying raw blind-set predictions, the same predictions after placement, and the blind population curve. CYP2C9 and CYP3A4 barely move. CYP1A2 shifts down and widens. CYP2D6 moves dramatically: a narrow spike at 4.5 with standard deviation 0.49 becomes a broad distribution centred at 3.1 matching the population curve." style="width: 100%; height: auto; display: block;">

CYP2D6 shows the failure at full scale: raw predictions are a spike at 4.5 with sd 0.49 against a population centred at 3.1 with sd 1.60. The model never called a single compound a non-inhibitor, and was predicting into less than a third of the actual range.

Placement is the whole of the difference between our worst and best submissions — same weights, same predictions, same ordering of every compound. Spearman and Kendall come back bit-identical under an affine transform, which is also the integrity check: if they move, the bug is in the submission pipeline.

**The catch.** Placement that maximises R² does not minimise the scored metric. Soft-threshold RAE is zero anywhere inside a compound's credible interval, and low-activity compounds — below the lowest tested dose — carry wide intervals, so predicting high is nearly free while predicting low is punished by the actives. Placing CYP2D6 on its true centre raised R² from 0.363 to 0.447 and *worsened* ST-RAE from 0.565 to 0.694. Where the ST-RAE optimum actually sits is the open question we find most interesting.

## What Didn't Help

**Censored labels via bounded loss.** ~2,900 compounds showed no CYP2D6 inhibition at 50 µM — real measurements, not missing data. Bounded loss has no gradient below the bound, so the cheapest way to satisfy 2,627 rows bounded at one value is a constant just under it. That is what happened (σ 0.07), and it propagated: blind-set CYP2D6 predictions got **narrower** (σ 0.41 → 0.30), not wider. The root cause is the assay — CYP2D6's single-concentration response is flat across the low end where CYP3A4's is monotone. It can say "not an inhibitor" but not how far below.

**Architecture changes.** Single-task descriptor models tie the multi-task network on CYP2D6 (holdout Pearson 0.414 vs 0.419) while losing on the other three. Four primary-weighted variants tie the symmetric model at four times the cost.

**More data, in general.** CYP2C9 reaches Pearson 0.666 on fewer rows than CYP2D6 manages 0.398 with. Volume was never the constraint.

**Treating "no fitted curve" as "inactive."** Of CYP3A4's 2,570 unfitted compounds, 1,511 inhibited hard and simply failed to fit. CYP2D6 has zero such rows, which is why the rule looks safe if you check it there first.

**Hyperparameter search.** Gains on in-distribution cross-validation reverse out of distribution. Stock defaults remain the baseline to beat.

## Still To Come

Ranking is the remaining gap, concentrated in CYP2D6, where our Spearman of 0.468 trails the best entry's 0.561 while the other three sit within 0.025 of the best. Queued: averaging across random seeds, measured to add 0.024–0.044 Spearman on every isoform; the PubChem qHTS panel, which adds ~16,000 molecules of essentially disjoint chemistry and agrees with the challenge assay best on CYP2D6; and a sweep of the auxiliary-target weight. No TDI entry yet.

## Reproducing This

**Source code:** [`ml_pipelines/OpenADMET/cyp`](https://github.com/SuperCowPowers/workbench/tree/main/ml_pipelines/OpenADMET/cyp) — every script referenced here, including the failed censored-label experiment with its measurements in the docstring.

Built with [ADMET Workbench](https://github.com/SuperCowPowers/workbench):

```bash
# FeatureSets from the challenge release, then the auxiliary-target FeatureSet
python ml_pipelines/OpenADMET/cyp/cyp_feature_sets.py
python ml_pipelines/OpenADMET/cyp/cyp_aux_features.py

# The submitted model — 8 heads, trained on 100% of the data
python ml_pipelines/OpenADMET/cyp/cyp_chemprop_mt_aux_100.py

# Predict the 750 blinded compounds, then place them on the target population
python ml_pipelines/OpenADMET/cyp/scripts/cyp_submit.py
python ml_pipelines/OpenADMET/cyp/scripts/cyp_recalibrate.py --solved
```

Submission files are checked with OpenADMET's own validator, vendored from their tutorial repository, so the gate before uploading is the same code the platform runs.

## References

- [Our CYP pipeline source](https://github.com/SuperCowPowers/workbench/tree/main/ml_pipelines/OpenADMET/cyp)
- [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge)
- [CYP Challenge Tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial) — baseline notebooks, scoring harness, submission validators
- [A Weekend on the OpenADMET PXR Challenge](pxr_weekend_experiments.md) — our prior blind-challenge write-up, and where the HPO and 3D-descriptor priors come from
- van Tilborg et al., *Exposing the Limitations of Molecular Machine Learning with Activity Cliffs*, J. Chem. Inf. Model. 2022

## Questions?

Open an issue on [GitHub](https://github.com/SuperCowPowers/workbench) or reach us at [SuperCowPowers](https://www.supercowpowers.com).
