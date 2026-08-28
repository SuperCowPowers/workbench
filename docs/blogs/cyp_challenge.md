# OpenADMET CYP Challenge — Method Report

!!! tip inline end "Current entry"
    One multi-task Chemprop model over the challenge data plus public CYP potency, stock hyperparameters, no ensembling — plus a per-isoform placement correction. **Macro ST-RAE 0.4538, rank 1 of 80** as of 2026-08-28.

This is the standing method report for our entry to the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge) (Direct Inhibition track). It is updated as the entry changes; it always describes what is currently submitted.

One thing dominated everything else we tried: the training labels and the blind set are drawn from different populations, and correcting for that is worth more than any modelling change we made.

## The Model

A single multi-task Chemprop D-MPNN with eighteen heads on one shared encoder. Four are the scored pIC50 targets; the other fourteen exist only to supervise the representation.

| | |
|---|---|
| **Architecture** | Chemprop D-MPNN, one shared encoder, 18 task heads |
| **Features** | SMILES only — the learned graph representation, no descriptors |
| **Scored targets** | pIC50 for CYP1A2, CYP2C9, CYP2D6, CYP3A4 |
| **Auxiliary targets** | Single-concentration log2 fold-change ×4; public ChEMBL pIC50 ×5; public qHTS max-response ×5 |
| **Training data** | 31,670 compounds / 161,400 labels — the challenge release plus ChEMBL and the NCATS qHTS panel |
| **Task weights** | Inverse task counts on the scored targets; auxiliaries at a fixed fraction of their mean |
| **Hyperparameters** | Chemprop defaults |

**Why multi-task.** The isoforms are correlated and the per-isoform data is small — roughly 1,300–2,300 curves each. One encoder learning from all 6,525 measurements is a data-efficiency argument, not a claim that graph learning resolves activity cliffs; on cliffs, descriptor models match or beat GNNs. OpenADMET report the same finding in their own tutorial.

**Why auxiliary heads rather than more rows.** Every extra source measures something adjacent to the scored target but not identical to it. The challenge's single-concentration arm reports a fold-change, not a pIC50. ChEMBL's potency comes from other labs and reads about 0.5 log more potent than the challenge assay on shared compounds. The qHTS panel reports efficacy at the top concentration, on a different scale again. Pooling any of them into the scored columns would need a cross-assay correction we would have to get right; as separate heads it is unnecessary — the encoder learns from all of them and each head keeps its own scale.

That also sidesteps what killed our censored-label attempt below. A readout that cannot say *how far* below a threshold a compound sits is useless as a bound and perfectly usable as a target.

**What the public data bought.** ChEMBL contributes ~24,700 compounds of chemistry disjoint from the challenge deck (185 structures overlap; **zero** overlap with the blind set). The qHTS panel contributes max-response, which is recorded for every compound whether or not it inhibited — so the ~42,000 rows that showed nothing carry signal, where a potency-only source drops them.

Against the same model without it, CYP1A2's ranking on the blind set improves enough to take that isoform to first place on its leaderboard. That is the only modelling change we have been able to confirm against the real test set rather than against our own cross-validation — roughly twenty-five thousand compounds of chemistry the challenge never touched, improving how a held-out screening deck gets ordered.

The other three isoforms moved by less than the leaderboard can resolve, which is a statement about the board's resolution rather than about the gain: two of the three moved in the right direction, and none of them cleared the noise in either direction.

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

<img src="../../images/cyp_calibration_applied.svg" alt="Four panels, one per isoform, each overlaying raw blind-set predictions, the same predictions after placement, and the blind population curve. CYP1A2, CYP2C9 and CYP3A4 shift and widen onto their population curves. CYP2D6 also moves down and widens but deliberately stops short, sitting above the population centre and narrower than it, because that is where the scored metric is optimised rather than R-squared." style="width: 100%; height: auto; display: block;">

CYP2D6 shows the failure at full scale: raw predictions are a spike at 4.5 with sd 0.49 against a population centred at 3.1 with sd 1.60. The model never called a single compound a non-inhibitor, and was predicting into less than a third of the actual range.

Placement is the whole of the difference between our worst and best submissions — same weights, same predictions, same ordering of every compound. Spearman and Kendall come back bit-identical under an affine transform, which is also the integrity check: if they move, the bug is in the submission pipeline.

**The catch.** Placement that maximises R² does not minimise the scored metric. Soft-threshold RAE is zero anywhere inside a compound's credible interval, and low-activity compounds — below the lowest tested dose — carry wide intervals, so predicting high is nearly free while predicting low is punished by the actives. Placing CYP2D6 on its true centre raises R² and *worsens* ST-RAE — on that isoform the two objectives point in opposite directions. Where the ST-RAE optimum actually sits is the open question we find most interesting.

## What Didn't Help

**Censored labels via bounded loss.** ~2,900 compounds showed no CYP2D6 inhibition at 50 µM — real measurements, not missing data. Bounded loss has no gradient below the bound, so the cheapest way to satisfy 2,627 rows bounded at one value is a constant just under it. That is what happened, and because those compounds span a huge swath of chemical space the constant propagated: blind-set CYP2D6 predictions came out **narrower**, not wider. The root cause is the assay — CYP2D6's single-concentration response is flat across the low end where CYP3A4's is monotone. It can say "not an inhibitor" but not how far below.

**Architecture changes.** Single-task descriptor models tie the multi-task network on CYP2D6 while losing clearly on the other three, so shared-encoder interference was never what suppressed it. Four primary-weighted variants tie the symmetric model at four times the cost.

**More data, in general.** CYP2C9 is our best-predicted isoform on the fewest training rows, and CYP2D6 the worst on more. Volume was never the constraint.

**Treating "no fitted curve" as "inactive."** Of CYP3A4's 2,570 unfitted compounds, 1,511 inhibited hard and simply failed to fit. CYP2D6 has zero such rows, which is why the rule looks safe if you check it there first.

**Hyperparameter search.** Gains on in-distribution cross-validation reverse out of distribution. Stock defaults remain the baseline to beat.

## Still To Come

**CYP2D6 is the gap.** Spearman 0.443 against a field best near 0.56, where the other three isoforms sit at or near the top of the board. It is the isoform the challenge did not select hits on, so its blind population is the lowest-centred and widest of the four, and nothing we have tried — censored labels, architecture changes, extra public data — has moved it.

**A measurement problem we did not expect.** Out-of-fold cross-validation on the training set cannot resolve the differences we are now chasing — repeated training runs of one configuration at different random seeds disagree by more than our candidate models do. The leaderboard is not much better: the standard error on a Spearman is largest where the correlation is weakest, so CYP2D6, the isoform we most need to improve, is the worst-resolved on both rulers.

Measuring the noise floor before running experiments is the cheapest hour in the project, and we ran it late. Most of what we would otherwise have "learned" from small differences was run-to-run variance.

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
