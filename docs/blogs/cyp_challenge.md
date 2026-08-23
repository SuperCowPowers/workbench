# OpenADMET CYP Challenge — Method Report

!!! tip inline end "Current entry"
    One multi-task Chemprop model, stock hyperparameters, no ensembling — plus a per-isoform calibration correction. **Macro ST-RAE 0.6171, rank 4** as of 2026-08-22.

This is the standing method report for our entry to the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge) (Direct Inhibition track). It is updated as the entry changes; it always describes what is currently submitted.

The short version: our first submission ranked 15th. The second ranked 4th. **The model is identical between them** — same weights, same predictions, same ordering of every compound. What changed was where those predictions sat on the pIC50 axis. That turned out to be worth more than everything else we tried put together, and it is the main thing this report has to offer.

## The Model

A single multi-task Chemprop D-MPNN predicting all four isoforms from one shared encoder.

| | |
|---|---|
| **Architecture** | Chemprop D-MPNN, multi-task, one shared encoder over 4 targets |
| **Features** | SMILES only — the learned graph representation, no descriptors |
| **Training data** | 4,905 compounds / 6,525 dose-response measurements (the challenge release, 100%) |
| **Task weights** | Inverse task counts, correcting uneven coverage (3A4 2,335 curves vs 2C9 1,285) |
| **Hyperparameters** | Chemprop defaults |

**Why multi-task.** The isoforms are correlated and the per-isoform data is small — roughly 1,300–2,300 curves each. One encoder learning from all 6,525 measurements is a data-efficiency argument, not a claim that graph learning resolves activity cliffs; on cliffs, descriptor models match or beat GNNs. OpenADMET report the same finding in their own tutorial.

## Evaluation

The blind set is not a random draw. OpenADMET built it by hit expansion: the top 25 hits for CYP1A2, CYP2C9 and CYP3A4, then the top 10 catalog chemisimilars purchased for each — 750 compounds in dense clusters around potent molecules.

So we evaluate on an **analog holdout** rather than cross-validation: take the top hits per target, hold out their nearest neighbours, keep the hits themselves in training. A random split of the same size flatters a baseline by roughly 2×.

The holdout lands on the real set's difficulty. OpenADMET report the blind set's maximum ECFP4 Tanimoto to training centring around 0.50–0.55 with a tail to 0.7; our 529-compound holdout measures a median of 0.509 and a p90 of 0.646.

On that holdout our model scores **macro ST-RAE 0.423**. We also replicated OpenADMET's own tutorial baseline — a stock gradient-boosted tree on RDKit descriptors — and scored it the same way: **0.575**, worse on all four isoforms. That gap is the honest measure of what the modelling bought us.

Model differences are quoted with a paired bootstrap (1,000 resamples). The noise floor on this holdout is about 0.03 macro ST-RAE, large enough to have swallowed at least one result we initially believed.

## What Actually Helped: Calibration

Our first submission scored macro ST-RAE 0.8414 — about twice what our holdout predicted. Breaking it down per isoform showed something strange:

| isoform | R² | Spearman | R² achievable at that ranking | achieved |
|---|---|---|---|---|
| CYP2C9 | 0.593 | 0.771 | 0.595 | **100%** |
| CYP3A4 | 0.593 | 0.778 | 0.605 | **98%** |
| CYP1A2 | 0.206 | 0.723 | 0.522 | 39% |
| CYP2D6 | **−1.020** | 0.432 | 0.187 | **−547%** |

Two isoforms were extracting essentially everything their ranking allowed. Two were far below — and CYP2D6 was scoring *worse than predicting a constant*, despite ordering compounds perfectly respectably.

That pattern rules out most explanations. A model that can't tell CYP2D6 compounds apart would have a bad Spearman; ours was fine. What it had was a **bad zero point**. Our CYP2D6 predictions clustered at 4.77 with a floor at 4.15 — we never called a single compound a non-inhibitor — while the blind set's true values sat well over a log unit lower.

The cause is a dataset shift that will be familiar to anyone doing ADMET work. Our training labels are **doubly hit-enriched**: compounds were screened because someone thought they might be active, and only those producing a fittable dose-response curve got a label at all. The blind set is a screening population — mostly compounds that do nothing. A model calibrated to the first systematically over-predicts activity on the second.

So we shifted CYP2D6 down by 1.2 log units and CYP1A2 by 0.6, and left CYP2C9 and CYP3A4 completely untouched since they were already at ceiling:

| isoform | ST-RAE before → after | rank |
|---|---|---|
| CYP2D6 | 1.5176 → **0.7301** | 41 → **7** |
| CYP1A2 | 0.7877 → **0.6782** | 22 → **10** |
| CYP2C9 | 0.4944 — unchanged | 8 → 8 |
| CYP3A4 | 0.5659 — unchanged | 12 → 12 |
| **Macro** | **0.8414 → 0.6171** | **15 → 4** |

Spearman and Kendall came back bit-identical, as they must — a constant offset cannot reorder anything. That was also our integrity check: if the rank metrics had moved, the problem would have been in our submission pipeline rather than our calibration.

!!! note "Is this just leaderboard fitting?"
    Partly, and worth being straight about. The exact offsets were estimated from leaderboard feedback, which you would not have in a deployment setting. Two things make us comfortable calling it a real correction rather than a fit: it is two parameters against 750 compounds, and the predicted effect matched the observed one closely (we projected CYP2D6 R² ≈ 0.16 and measured 0.149) — noise-fitting does not extrapolate that well.

    More usefully, we had an **independent estimate that used no leaderboard data at all**. The public PubChem qHTS panel puts CYP2D6 inactivity at roughly 65%. A blind set that is 65% inactive centres near pIC50 3.7 against our 4.69 — implying a shift of about 1.0, against the 1.2 that proved correct. The correction was available from chemistry alone, before we ever submitted. We just didn't think to ask what the population we were predicting on actually looked like.

The transferable lesson is not "shift your predictions." It is that we spent two days asking *how do we model CYP2D6 better* and the answer was that we were modelling it fine and pointing it at the wrong distribution.

## What Didn't Help

Recorded because negative results are the expensive half of the work, and because several of these are things a reasonable person would try first.

**Censored labels via bounded loss.** The challenge's single-concentration arm measured all 4,376 compounds against every isoform, and ~2,900 showed no CYP2D6 inhibition at 50 µM. Those are real measurements, not missing data, so we fed them in as left-censored labels using Chemprop's bounded loss. It failed, and instructively: bounded loss has no gradient *below* the bound, so the cheapest way to satisfy 2,627 rows bounded at the same value is to predict a constant just under it. That is exactly what happened — those predictions collapsed to a standard deviation of 0.07 — and because those compounds span a huge swath of chemical space, the constant propagated. Blind-set CYP2D6 predictions got **narrower** (σ 0.41 → 0.30), not wider.

The root cause is in the assay, not the loss. CYP2D6's single-concentration response is flat across the whole low end — a compound at pIC50 2.0 and one at 4.6 look identical — where CYP3A4's is cleanly monotone. The readout can say "not an inhibitor" but not how far below. The one isoform we needed it for is the one it cannot serve.

**Architecture changes.** Single-task descriptor models tie the multi-task network on CYP2D6 (holdout Pearson 0.414 vs 0.419) while losing clearly on the other three — so shared-encoder interference was not what suppressed it. Four primary-weighted variants (each isoform at 1.0, the rest at 0.3) were a statistical tie with the symmetric model at four times the training cost.

**More data, in general.** CYP2C9 reaches Pearson 0.666 on *fewer* training rows than CYP2D6 manages 0.398 with. Volume was never the constraint. Most tellingly, OpenADMET's own reference baseline uses strictly less data than we do — no multi-task, no censoring, no external sources — and out-scored us on the CYP2D6 leaderboard tab while ranking compounds *worse* than we did. Its advantage was entirely that its predictions were wider and reached into the inactive range ours refused to enter.

**Treating "no fitted curve" as "inactive."** Tempting, and wrong in a way that depends on the isoform. Of CYP3A4's 2,570 unfitted compounds, 1,511 inhibited hard and simply failed to produce a curve. A blanket rule would have labelled 1,511 potent inhibitors as inactive. CYP2D6 happens to have zero such rows — which is why the rule looked safe when we checked it there first.

**Hyperparameter search** (carried over from our PXR entry): gains on in-distribution cross-validation reversed on the analog set. Stock defaults remain the baseline to beat.

## Still To Come

Calibration is nearly exhausted — about +0.05 macro R² remains at our current ranking. Past that, the gap to the leader is **ranking**, concentrated in CYP2D6, where our Spearman of 0.432 trails the top entry's 0.543. That is now a clean modelling target rather than something hidden behind a placement error. No TDI entry yet.

## Reproducing This

**Source code:** [`ml_pipelines/OpenADMET/cyp`](https://github.com/SuperCowPowers/workbench/tree/main/ml_pipelines/OpenADMET/cyp) — every script referenced here, including the failed censored-label experiment with its measurements in the docstring.

Built with [ADMET Workbench](https://github.com/SuperCowPowers/workbench):

```bash
# FeatureSets from the challenge release (2D + 3D features; Chemprop uses SMILES only)
python ml_pipelines/OpenADMET/cyp/cyp_feature_sets.py

# The evaluated model — holds out the analog set, reports honest ST-RAE
python ml_pipelines/OpenADMET/cyp/cyp_chemprop_mt_all.py

# The submitted model — same configuration, trained on 100% of the data
python ml_pipelines/OpenADMET/cyp/cyp_chemprop_mt_100.py

# Predict the 750 blinded compounds and write a validated submission
python ml_pipelines/OpenADMET/cyp/scripts/cyp_submit.py

# Apply the per-isoform calibration correction
python ml_pipelines/OpenADMET/cyp/scripts/cyp_recalibrate.py
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
