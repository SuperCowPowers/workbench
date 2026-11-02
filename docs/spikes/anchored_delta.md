# Anchored-Delta Correction on PXR

!!! note inline end "Verdict: negative"
    Correcting predictions with nearest-analog residuals does not improve held-out PXR error for Chemprop or TabICL + Monroe. Every paired 95% interval includes zero.

[Causal-Chemprop](https://github.com/neopolyai/causal_chemprop) reports gains on kinase pIC50 (ABL1, AURKA, SRC) from counterfactual inference. It fits a structural causal model (SCM) over Chemprop's latent dimensions plus pIC50, infers a measured anchor compound's noise term, then swaps the high-influence latent dimensions for the target's. Because the anchor's noise term is kept, the prediction amounts to the anchor's measured value plus a predicted change. This spike tests that anchoring core in its simplest form, on a larger dataset.

## Method

$$
\hat{y}(x) = \text{pred}(x) + \alpha \cdot \frac{\sum_{j \in \text{top-}k} s_j \, r_j}{\sum_{j \in \text{top-}k} s_j}
$$

$s_j$ is Tanimoto similarity (sparse count Morgan, radius 2) to anchor $j$, and $r_j = y_j - \text{pred}_j$ is its residual. With $k=1, \alpha=1$ this is exactly the anchored baseline $\text{pred}(x) - \text{pred}(a) + y(a)$.

| | |
|---|---|
| **Data** | `openadmet_pxr_f1`: 4,139 training compounds; 253 held-out Analog Set 1 (`pxr_phase1_test` capture) |
| **Models** | `pxr-reg-chemprop-phase1`, `pxr-reg-tabicl-monroe-readout-phase1` (5-fold scaffold CV) |
| **Selection** | $k \in \{1,3,5,10,25\}$, $\alpha \in \{0.25,0.5,0.75,1.0\}$, chosen on training OOF by leave-one-out over the anchor pool, then applied once to the held-out set |
| **Train anchors** | OOF residuals of the training compounds (usable for a blind submission) |
| **Series anchors** | The other held-out compounds' residuals, leave-one-out (closest to the paper's setting: part of the new series is already measured) |
| **Test-tuned** | $(k, \alpha)$ swept on the held-out set itself; an optimistic ceiling, not a result |

`resid_corr` is the Spearman correlation between a compound's residual and the correction its neighbors supply. It is the direct test of whether there is local structure to transfer.

## Results

Held-out RAE (lower is better), deltas paired-bootstrapped against the baseline.

| Arm | Chemprop RAE | Δ [95% CI] | resid_corr | TabICL + Monroe RAE | Δ [95% CI] | resid_corr |
|---|---|---|---|---|---|---|
| Baseline | 0.591 | | | 0.535 | | |
| Train anchors | 0.592 | +0.001 [−0.002, +0.005] | −0.03 | 0.528 | −0.007 [−0.017, +0.002] | 0.10 |
| Series anchors | 0.585 | −0.005 [−0.014, +0.003] | 0.17 | 0.530 | −0.005 [−0.013, +0.004] | 0.15 |
| Test-tuned, train | 0.589 | | | 0.528 | | |
| Test-tuned, series | 0.575 | | | 0.525 | | |

On training OOF the selected corrections are $k=25, \alpha=0.25$ for Chemprop (RAE 0.540 → 0.540, resid_corr 0.02) and $k=10, \alpha=0.25$ for TabICL + Monroe (0.505 → 0.502, resid_corr 0.11).

Held-out MAE by similarity to the nearest training compound:

| Max similarity | n | Chemprop base | Chemprop corrected | TabICL base | TabICL corrected |
|---|---|---|---|---|---|
| < 0.4 | 6 | 0.562 | 0.581 | 0.522 | 0.521 |
| 0.4 – 0.6 | 195 | 0.483 | 0.485 | 0.433 | 0.428 |
| ≥ 0.6 | 52 | 0.418 | 0.414 | 0.397 | 0.387 |

## Interpretation

- **Chemprop's residuals carry no neighbor structure** (resid_corr 0.02 on training OOF). With about 4k compounds the model has already absorbed the local SAR, so a neighbor's error says nothing about the next compound's.
- **TabICL + Monroe has a little more** (0.11), plausibly because it learns on a fixed embedding rather than end to end. The gain it buys is small and inside the noise.
- **Even series anchors don't help.** Using measured members of the held-out series gives at most −0.005 at the train-selected setting, and −0.016 when tuned on the test set itself.

The regime where anchoring should pay is one with structured local error: small training sets and test analogs close to a measured lead, as in the paper's kinase sets. PXR is a fair test of whether this helps at Workbench's data scale; it is a weak test of the paper's claim.

## What This Does Not Test

- **Damped delta.** The partial intervention behaves like $y(a) + \beta\,(\text{pred}(x) - \text{pred}(a))$ with $\beta < 1$, shrinking the predicted change toward the anchor. Our $\alpha$ scales the transferred residual instead, a different family.
- **Latent-space similarity** and intervention on selected latent dimensions; we use Tanimoto neighbors.
- **The single-lead, small-data setting.** The direct test of the paper's claim is the anchored baseline on its own Zenodo data and checkpoints (pinned to chemprop 2.0.0), which separates the SCM's contribution from the anchor label's.

## Open Questions on the Reference Repo

From reading the code, not the paper:

- The Chemprop baseline does not receive the anchor's measured value, while the counterfactual prediction does.
- Each prediction chooses between two outputs using a per-dataset `THRESHOLD`, and `icc_threshold` is set from test-set similarity; we found no separate tuning split.
- The repo has no license, so the code cannot be reused.

## Reproduce

```
python spikes/anchored_delta/pxr_anchored_delta.py
```

Reads the two models' `full_cross_fold` and `pxr_phase1_test` captures; runs in under a minute. The Chemprop `full_cross_fold` capture also scores the held-out rows, so the script drops them from the anchor pool.
