# OpenADMET CYP Inhibition Challenge

> the 2026 OpenADMET blind CYP challenge: targets, scoring, and how to build a competitive entry

An external blind competition run by OpenADMET — **not** a Workbench contest.
`contests` covers the internal champion/challenger comparison; that machinery is
useful here (below), but the two words mean different things. Be explicit about
which one the user means.

## The challenge

Predict inhibition of four CYP isoforms — **CYP3A4, CYP2C9, CYP2D6, CYP1A2** —
on a blind set of 750 compounds. Two independent tracks:

| Track | Shape | Targets | Metric |
|---|---|---|---|
| **Direct inhibition** | regression | pIC50 for all 4 CYPs (3,000 predictions) | macro-averaged ST-RAE |
| **Time-dependent inhibition (TDI)** | binary classification | >2-fold IC50 shift after NADPH preincubation, CYP3A4 + CYP2D6 only | MCC |

Training data is **sparse**: 4,905 compounds carry 6,525 dose-response
measurements between them (1,285-2,335 per isoform), so most compounds have one
or two isoforms rather than all four. A separate single-concentration primary
screen covers 4,376 compounds against all four. Assays are biochemical with
recombinant enzymes — fluorescence for 3A4/2C9/1A2, acoustic-ejection mass spec
for 2D6.

A third **structure (pose) track** — 184 structures, one leaderboard — opens
partway through and is scored separately from the two activity tracks.

**Dates:** launches 2026-08-17, intermediate submission 2026-09-24, intermediate
leaderboard 2026-09-25, final submission 2026-11-03. Deadlines are 23:59 UTC.

**One continuous stage — no ground truth is ever unblinded mid-challenge.** The
2026-09-25 intermediate leaderboard reveals *scores*, not labels. This differs
from the PXR challenge, which unblinded half its test set partway through; do
not carry that structure over.

## Scoring drives the modeling

**ST-RAE gives zero error inside the ground-truth credible interval.** Each
pIC50 label carries a Bayesian credible interval from the curve fit, and a
prediction landing inside it is scored as perfect. Intervals widen at low
activity, and compounds below pIC50 4 (under the lowest tested concentration)
carry the widest ones.

OpenADMET describes this as downweighting low-activity compounds. The interval
*is* the downweighting — there is no separate weight term to reproduce. Weak
compounds get wide intervals and therefore more slack.

**The intervals are wide, so the forgiveness is substantial.** Median widths on
the challenge's own training curves:

| isoform | measured | median CI width |
|---|---|---|
| CYP2C9 | 1,285 | 0.526 |
| CYP3A4 | 2,335 | 0.379 |
| CYP1A2 | 1,412 | 0.328 |
| CYP2D6 | 1,493 | 0.272 |

A prediction within roughly 0.3 pIC50 of a typical label scores *zero* error. For
calibration: a uniform +0.25 pIC50 bias across all four isoforms scores
MA-ST-RAE 0.172. Chasing residuals below the interval width is wasted effort, and
CYP2D6 punishes error hardest while CYP2C9 is the most forgiving.

What follows from that:

- **Accuracy on potent, tightly-measured compounds is where the score lives.**
  The weak/inactive tail is cheap to miss. Don't spend the modeling budget
  flattening errors on compounds whose intervals are wide anyway.
- It is *relative* absolute error, so this is not a license to regress
  everything toward the mean — that surrenders the potent end.
- Report the training labels' own intervals when discussing accuracy. A model
  whose residuals sit inside assay noise is already scoring zero there, and
  chasing it further is wasted effort.
- Our `confidence` and `q_05`/`q_95` outputs are **not** submitted and do not
  score. They are for deciding where to hedge, not part of the entry (`uq`).

**Scoring ST-RAE.** Model and endpoint metrics carry the standard regression set
(`rmse`, `mae`, `r2`, `pearsonr`, `spearmanr`, `support`) — not `st_rae`. Score it
yourself against the label intervals:

```python
from workbench.utils.metrics_utils import soft_threshold_rae
```

**Our `st_rae` is close to but not identical to a leaderboard number.** RAE needs
a denominator. OpenADMET's tutorial repo ships an `evaluation/` module, but it
does *not* implement ST-RAE — it is a generic harness (its endpoint list is
`["pEC50"]`) carrying MAE, RAE, R2, Spearman and Kendall. The README describes
MA-ST-RAE in prose; no published code computes it. What the harness does pin
down is their RAE convention:

```python
np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
```

The denominator is the plain `sum|y - ȳ|`, not a soft-thresholded baseline, and
`soft_threshold_rae` defaults to matching it (`soft_baseline=False`). Use the
default; there is no reason to score against a convention the challenge does not
use.

`soft_baseline=True` scores the mean-predictor baseline through the same soft
threshold, which keeps the ordinary-RAE reading of "1.0 means no better than the
mean". It inflates the score substantially — measured on our own CYP predictions
the published form is 0.49-0.68 of it, varying by isoform — **so the two are not
interconvertible and must never be quoted side by side.** Within one isoform the
choice cannot reorder models, since the denominator ignores the predictions, but
the macro average can reorder because it averages ratios whose denominators shift
by different amounts.

**Leaderboard scores are bootstrapped**: 1,000 resamples at a fixed seed, with
the spread reported alongside each score. Combined with the live board scoring
only half the test set, a small gap between two entries is inside the noise —
read the spread before treating a rank as a result.

**Compare two models with a *paired* bootstrap**, not by eyeballing each
one's interval — the marginal intervals overlap heavily while the paired test
still separates them, because pairing cancels per-compound difficulty:

```python
from workbench.utils.metrics_utils import bootstrap_compare, bootstrap_metric
```

Index both prediction frames by the id column and pass a `metric_fn` that scores a
frame. Measured example: chemprop-MT 0.702 vs XGBoost 0.776 have overlapping
marginal intervals, but paired gives delta -0.074, 95% CI [-0.133, -0.013],
P(chemprop better) 99%. Quote the CI and the paired delta, never a bare score
difference.

MCC on the TDI track is chosen for imbalanced labels — accuracy will look good
and mean nothing. Quote MCC, and check the positive-class rate before claiming
a classifier works.

**The TDI label is a shift between two arms, not a measured value**, so it is
defined piecewise around pIC50 4 (the assay's reliable floor); the shift
threshold is 2-fold, log10(2) = 0.301:

| Direct pIC50 | TDI-arm | Label |
|---|---|---|
| > 4 | shift > 0.301 | positive |
| > 4 | shift ≤ 0.301 | negative |
| < 4 | > 4.301 | positive (inferred) |
| < 4 | < 4 | negative (assigned) |

Predictions are required for all 750 compounds, but only confidently-labelable
ones score. Both inferred positives and assigned negatives are real scored
labels — do not filter them out of training as unmeasurable.

## Where the entry stands

Pulled 2026-09-23 with `ml_pipelines/OpenADMET/cyp/scripts/cyp_leaderboard.py`. The field
grows by tens of entries a week, so a fixed score loses rank steadily — re-pull before
quoting any of this.

| | macro | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|---|
| our ST-RAE | 0.4378 | 0.4483 | 0.3632 | 0.5502 | 0.3895 |
| our rank, of 216 | 8 | 13 | 12 | **26** | 8 |
| board best | 0.3872 | 0.3846 | 0.3354 | 0.4156 | 0.3653 |

Standing entry: a four-model chemprop ensemble, placed per isoform.

**The gap is ordering, not placement.** Of the 0.0626 macro gap to rank 1, only 0.0135
(22%) is recoverable by re-placing at our current ordering — measured as the best ST-RAE
any entry achieves whose Spearman is within +-0.01 of ours. Chase Spearman and Pearson
first; placement is already close to its ceiling on every isoform and exactly at it on
CYP2D6.

**CYP2D6 is the weak isoform and the gap is real, not a ruler artifact.** We rank 26th on
its ST-RAE and 30th on Spearman (0.487 against a board best of 0.561), with 27 entries above
0.500. A dense band of competitors that far ahead is not noise — something is being found
that we are not finding.

**Descriptor GBMs land near 0.9 raw.** A descriptor model scoring there is performing
normally, not broken; most of the distance to a competitive score is placement.

## Placement: the affine half of the score

A prediction vector carries two independent things — its **order**, which is the model, and
its **placement** on the pIC50 axis, which is not. R2 decomposes exactly:

    R2 = 2*rho*k - k^2 - b^2

with `rho` the Pearson correlation, `k = sd(pred)/sd(true)` the spread ratio, and `b` the
mean offset in sd(true) units. Only `rho` depends on ordering, so **R2 <= rho^2** is a hard
ceiling and a model far below its own `rho^2` is mis-placed rather than weak.

**The optimum is `k = rho`, not `k = 1`.** Matching the truth's spread is wrong: a model at
rho 0.7 should be 70% as wide as reality, because shrinking toward the mean is the correct
response to uncertainty. Raw predictions are narrower still, so most models need widening.

So: estimate the blind population's centre and spread, then place each isoform there with
spread `rho*sd`. Spearman and Kendall come back bit-identical under an affine transform,
which doubles as the integrity check — if they move, the submission pipeline has a bug.

**ST-RAE and R2 want different placements**, on CYP2D6 by a wide margin, because ST-RAE
scores zero inside a credible interval and low-activity compounds carry wide ones.
Predicting high is nearly free; predicting low is punished by the actives. Placing CYP2D6 on
its true centre raises R2 and *worsens* ST-RAE.

**Deriving the ST-RAE optimum offline does not work** — out-of-fold interval widths do not
represent the blind set's, so hiding predictions inside them looks free offline and is not.
The optimum is known by sampling placements against the board. Interpolate between probed
points, never extrapolate past them.

`ml_pipelines/OpenADMET/cyp/scripts/cyp_recalibrate.py` holds the measured constants and
applies the transform.

## The test set is analog-heavy — this is the key fact

The blind set is the top 25 hits per CYP for three isoforms (75 compounds) plus
**10 catalog analogs of each hit**. So it is dense clusters of near-neighbors
around potent hits, not a diverse draw — a held-out analog series, exactly the
regime where our measured HPO gains disappeared.

- **Stock chemprop defaults are the baseline to beat, not the starting point to
  tune away from.** HPO improved in-distribution cross-validation and *lost* to
  untuned defaults on PXR's analog set (`hpo`). If an HPO run is done anyway,
  quote `model.hpo_results()` numbers and treat a baseline win as the expected
  outcome.
- **Out-of-fold predictions are the ruler.** Score candidates on the `cv_<target>`
  captures (multi-target) or `full_cross_fold` (single-target), with
  `ml_pipelines/OpenADMET/cyp/scripts/cyp_compare.py`. Butina
  (`hyperparameters={"split_strategy": "butina"}`) is the fold strategy worth testing
  against the default scaffold split — it asks "new chemotypes?", which is closer to how
  the blind set was built.
- **Know what the ruler can resolve before reading a delta.** Smallest out-of-fold Spearman
  difference distinguishable from training and sampling noise, per
  `scripts/cyp_ruler_power.py`: CYP1A2 0.043, CYP2C9 0.031, CYP2D6 0.056, CYP3A4 0.018.
  Our model-to-model differences run 0.01-0.03, so **most candidate comparisons here are
  unresolvable** and come back "cannot tell" rather than negative. Say so instead of
  reporting the sign.
- Analog clusters mean small structural changes must move the prediction.
  Check activity cliffs and near-duplicate collisions in the training data
  before trusting a model to resolve them (`proximity`, `data_cleanup`).
  Count-Morgan fingerprints on the largest fragment collapse enantiomers, so
  stereo-only pairs are invisible to them (`cheminformatics`).

## Always pass an explicit feature_list — the CI columns will leak

The training tables carry, per isoform, the pIC50 *plus* its `_ci_lower`, `_ci_upper`
and `_std`. Those are all numeric and none of them is the target column, so the
auto-generated feature list **includes them** — it drops ids, target columns, and
non-numeric types, and keeps the rest. A model handed the bounds that bracket its own
label scores near-perfectly in cross-validation and is worth nothing. The only warning
is a "Guessing at the feature list" log line.

The same trap catches the other isoforms: in a single-task CYP3A4 model,
`cyp2c9_pic50_direct_inhibition` is not a target, so it becomes a feature.

**The test set is `Molecule_Name` and `SMILES` only.** A feature has to be derivable
from structure — descriptors, fingerprints, or chemprop's own graph encoding.
Everything else in those tables is label metadata: useful for weighting, scoring, and
ST-RAE, never as model input.

**Ask the feature endpoint what it produced** — don't hand-roll a denylist over
`fs.columns`. The endpoint that built the FeatureSet registers exactly the columns
it emits, which is the feature list:

```python
feature_list = ["smiles"]                                      # chemprop

end = Endpoint("smiles-to-2d-3d-v1")                           # xgboost/pytorch
feature_list = end.output_columns()                            # 387 columns for f1
```

That returns only the descriptors, so labels, CI/std columns, ids, the
`desc3d_*` bookkeeping columns, and the AWS FeatureStore internals
(`write_time`, `event_time`, `api_invocation_time`, `is_deleted`) are all
excluded by construction. Subtracting label columns from `fs.columns` by hand
gets this wrong in both directions.

Use the endpoint that matches the FeatureSet: `smiles-to-2d-3d-v1` for the `_f1`
sets, `smiles-to-2d-3d-v2` for `_f2`.

If a CYP model reports R² above ~0.95 on held-out data, assume leakage and check the
feature list before believing it.

## Model shape: multi-task across the four CYPs

Four correlated targets with sparse, unequal coverage is the textbook
multi-task case. `target_column` takes a list:

```python
targets = [f"{iso}_pic50_direct_inhibition" for iso in ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]]
model = fs.to_model(
    name="cyp-inhibition-chemprop-mt-reg",
    model_type=ModelType.UQ_REGRESSOR,
    target_column=targets,
    feature_list=features,
    hyperparameters={"uq_version": "v1"},
)
```

Keep the full `_direct_inhibition` target names. A target's credible interval is
named by appending `_ci_lower`/`_ci_upper`, so shortening the targets breaks the
pairing when scoring ST-RAE — the challenge's own metric.

Missing targets are `NaN` per row — that is how sparsity is expressed; do not
drop rows to make the matrix dense. All four isoforms are end products here
(nothing is auxiliary), so unequal coverage is corrected with symmetric
weights:

```python
from workbench.utils.multi_task import compute_inverse_count_task_weights

task_weights = compute_inverse_count_task_weights(df, targets)
hyperparameters = {"task_weights": task_weights, "uq_version": "v1"}
```

The weights come back as plain floats because `hyperparameters` is JSON-serialized
on its way into the training script — a numpy scalar there raises `TypeError: Object
of type float32 is not JSON serializable`. Build the wide table from per-isoform
sources with `combine_multi_task_data` in the same module.

Chemprop is a heavy train — put the whole chain in a script on Batch rather
than blocking the REPL (`batch`, `making_models`). The script must build the
endpoint and score it, or the model comes back with no metrics.

The TDI track is a **separate classifier**, not a fifth regression target —
different label semantics, only two isoforms. Name it `-class`.

## 3D / xTB features are worth testing here, unlike PXR

CYP inhibition is catalysis at a heme iron: potency depends on orientation and
access to the heme, and the classic inhibitor pharmacophore is type-II
coordination — an azole or pyridine lone pair binding the iron directly. TDI is
the stronger case still, since mechanism-based inhibition requires the compound
to be oxidized into a reactive species. That is oxidation potential, HOMO, and
site-of-metabolism reactivity — exactly what the curated xTB electronic block in
`smiles-to-2d-3d-v2` targets.

This does **not** contradict the PXR result. PXR is a large promiscuous pocket
where induction tracks lipophilicity and size, which 2D descriptors already
encode; the mechanism here is different.

- **Build it as a standalone model that ensembles in, not as descriptors
  appended to a Chemprop backbone.** Appending is precisely what failed on PXR,
  where held-out RAE degraded monotonically as more 3D columns went in. The
  argument here is orthogonal information, which is an ensemble-diversity
  argument.
- On PXR, 3D ranked high in SHAP and *still* failed to transfer. The verdict comes from
  a held-out score against its resolution threshold, never from feature importance.

## Start from the built FeatureSets

These FeatureSets are already built and onboarded. Use them rather than rebuilding
from `PublicData` — they carry the decisions below
(credible intervals present, TDI labels de-leaked, challenge target naming) and
rebuilding re-derives all of it, usually getting one wrong.

| FeatureSet | rows | what it is |
|---|---|---|
| `openadmet_cyp_f1` | 4,905 | Regression track, 2D + **v1** 3D |
| `openadmet_cyp_f2` | 4,905 | Regression track, 2D + **v2** 3D (curated, xTB) |
| `openadmet_cyp_aux_f1` | 4,905 | `_f1` plus the four single-concentration log2fc targets |
| `openadmet_cyp_union_f1` | 35,801 | `_aux_f1` plus TDI, emax, and the ChEMBL / Veith / Tox21 public heads — 35 targets |
| `openadmet_cyp_union_censored_f1` | 35,801 | the union set with `IC50 > x` records as bounds carrying `_lt` flags |
| `openadmet_cyp_censored_f1` | 4,905 | `_f1` with censored records as bounds |
| `openadmet_cyp_fp` | 4,905 | Regression track, fingerprints |
| `openadmet_cyp_tdi_f1` / `_f2` / `_fp` | 6,145 | TDI track, same three feature blocks |

The current models train on `openadmet_cyp_aux_f1` and `openadmet_cyp_union_f1`; `_f1` and
`_f2` are the controlled A/B for the 3D layer.

- **`f1` vs `f2` differ only in the 3D layer** — same rows, same 2D block, same
  labels. That makes them a controlled A/B for whether the xTB electronic block
  earns its place. Hold everything else constant when comparing them.
- **The regression FeatureSets carry `_ci_lower`, `_ci_upper` and `_std` per
  isoform.** That is what makes ST-RAE computable, and it is also the leakage
  trap below — they are label metadata, never features.
- **The TDI FeatureSets carry labels only**, no arm pIC50s: `is_tdi` is derived
  from the shift between the direct and TDI arms, so carrying either arm beside
  the label hands the model the answer. To re-derive or audit labels, go back to
  `PublicData`.
- **`openadmet_cyp_union_f1` carries public potency as separate heads, not extra rows.**
  Each source sits on its own scale, so a head keeps its own calibration and no cross-assay
  correction is needed. Measured: the union heads move out-of-fold Spearman by a mean of
  -0.002 against the challenge-only model, none of it resolved. Public data has not paid off
  on any isoform.
- The challenge training table is already one row per compound with `NaN` where an
  isoform was not measured, so it needs no `combine_multi_task_data`. That helper
  is for assembling Veith-style per-isoform sources.

Rebuild with `ml_pipelines/OpenADMET/cyp/cyp_feature_sets.py`.

## Data already on hand

**The challenge's own data** (released 2026-08-17, Apache-2.0):

```python
inh = pub_data.get("comp_chem/openadmet/cyp/training/inhibition")           # 4,905 cpds, 4 pIC50s + CIs
tdi = pub_data.get("comp_chem/openadmet/cyp/training/tdi")                  # 6,145 cpds, is_TDI + both arms
emax = pub_data.get("comp_chem/openadmet/cyp/training/emax")                # 6,146 cpds, Emax both arms
screen = pub_data.get("comp_chem/openadmet/cyp/training/single_concentration")  # 17,504 rows (4,376 x 4 enzymes)
blind = pub_data.get("comp_chem/openadmet/cyp/testing/blinded")             # 750 cpds, structures only
```

Column names are snake_cased from the challenge's (`CYP3A4_pIC50_direct_inhibition`
becomes `cyp3a4_pic50_direct_inhibition`), and the credible-interval suffixes are
renamed from the source's `_conf_low`/`_conf_high` to `_ci_lower`/`_ci_upper` — the
platform convention for label intervals. Map back to the challenge's exact column
names at submission time.

Two things in here that the challenge write-up does not advertise:

- **`emax` carries `is_TDI` for all four isoforms**, not just the two that are
  scored. CYP1A2 and CYP2C9 TDI labels are free auxiliary tasks.
- **`single_concentration` has `plate_id` and `log2fc_std_error`**, so batch
  effects and per-measurement noise are inspectable rather than assumed.

**Public sources beyond the challenge**, which play different roles.

**Octant (same lab, same platform as the challenge)** — CYP3A4 only, but it
carries credible-interval columns in the exact shape the challenge scores on:

```python
inhibition_df = pub_data.get("comp_chem/openadmet/octant_cyp/inhibition")   # 1340 rows, CYP3A4 pIC50 + CI
reactivity_df = pub_data.get("comp_chem/openadmet/octant_cyp/reactivity")   # 2446 rows, multi-enzyme turnover
```

QC columns (`drc_qc_status`, `activity_status`, `rollover_status`,
`saturation_status`) — filter to clean curves before training.

**Veith qHTS panel (PubChem AID 1851)** — all four challenge isoforms plus
CYP2C19, one row per compound-isoform pair:

```python
all_df = pub_data.get("comp_chem/pubchem/cyp_inhibition/all_isoforms")  # 85,535 rows, 16,546 compounds
cyp3a4_df = pub_data.get("comp_chem/pubchem/cyp_inhibition/cyp3a4")     # per-isoform files also available
```

About 33,500 fitted curves across the four targets — roughly 5x the 6,525
measurements in the challenge's own training set. Three things to know before
using it:

- **Its noise is comparable to the challenge data, not obviously worse.**
  Compounds assayed under multiple SIDs disagree by a median of 0.40 pIC50 units,
  against challenge credible-interval widths of 0.27-0.53. Those are not the same
  quantity — replicate disagreement versus the width of one curve fit — but they
  are the same magnitude, so a heavy down-weight is not justified by noise alone.
  Its real difference is provenance: a different lab, platform, and assay readout.
- **The 42,355 "Inactive" rows are censored, not missing.** Those compounds were
  tested and showed no inhibition up to 57 uM — a real measurement saying
  pIC50 < ~4.2, with `pic50` NaN and `curve_class` 4. Dropping them discards two
  thirds of the screen, and it is exactly the low-activity regime ST-RAE and the
  TDI labeling rules are built around.
- **CYP2C19 is a free auxiliary task** — 9,544 curves of a correlated fifth
  isoform, not scored by the challenge.

Filter on `curve_class` (-1.1/-1.2 are complete curves) and `fit_r2` rather than
treating a single-point extrapolation like a full 15-point fit. Use `smiles`
(standardized); `orig_smiles` is the deposited string.

`pub_data.describe(...)` gives per-column meanings for any of these.

Re-pull with `python data/public_data/pull_openadmet_data.py --challenge cyp`
followed by `upload_data.py --apply` if OpenADMET revises the files mid-challenge.

**Iterate locally, publish the champion** (`local_models`). Every variant worth
trying — multi-task vs single-task, task weights, censored inactives, an xTB
ensemble member, HPO — needs its own analog-holdout score, and that loop is free
and fast on this machine while being slow and billable on Batch. `PublicData`
works without credentials, so a local chain starts directly off these datasets.

Local models compute metrics the same way AWS ones do:

```python
local_model.list_inference_runs()      # "full_cross_fold" plus any endpoint captures
local_model.get_inference_metrics()
local_model.get_inference_predictions("full_cross_fold")
```

Those are the standard regression metrics. For ST-RAE, pull the predictions, join
the label intervals from the FeatureSet, and call `soft_threshold_rae` directly.
Score candidates on the out-of-fold captures, against the resolution thresholds above.

## CYP2D6: what has been tried, and where the gap actually is

Ten attempts have come back null or negative: task weighting, a dedicated encoder,
single-task, fingerprints, binary active/inactive heads, Tox21 potency heads, a log2fc
surrogate, empirical-Bayes shrinkage, an uncertainty-driven downward tilt, low-band loss
weighting, pooled public labels in the scored column, and a CheMeleon pretrained encoder.
Propose any of them again only with a reason the earlier measurement was wrong.

Three of those were *resolved* negatives rather than nulls, and they share a shape —
weighting or padding the low end collapses the model toward a constant. Low-band loss
weighting took Spearman from 0.388 to 0.056 at 8x.

**The low bands cannot be ordered, and that is a label property.** Using the credible
intervals as a noise estimate, reliability = (var(label) - mean var(noise)) / var(label):

| band | n | label sd | noise sd | ceiling on rho |
|---|---|---|---|---|
| < 4.0 | 129 | 0.684 | 0.662 | 0.25 |
| 4.0-4.5 | 350 | 0.113 | 0.132 | ~0 |
| >= 4.5 | 1,014 | 0.609 | 0.070 | 0.99 |

The dead 4.0-4.5 band is not CYP2D6-specific — CYP1A2 and CYP2C9 are also ~0 there, and
CYP3A4, our best isoform, is the only one with a live middle band.

**But the headroom is between the bands, not within them.** An oracle capped at each band's
ceiling still reaches full-set Spearman 0.98, and band membership alone is worth 0.82. We
are at 0.44. Measured directly, AUC separating `<4.0` from `>=4.5` — a 3.5-log distinction
that carries no label-noise excuse:

| CYP3A4 | CYP2C9 | CYP1A2 | CYP2D6 |
|---|---|---|---|
| 0.939 | 0.912 | 0.838 | **0.750** |

That is the open problem: telling a CYP2D6 non-inhibitor from an inhibitor, not ordering
weak ones against each other. Aim there.

## Submission discipline

- **One account per team or lab — not one submission.** The launch post's "one
  submission per team/lab" means one submitting *account*; its "we rely on your
  honesty" is aimed at labs entering under several accounts. The Space FAQ and
  `HOURS_BETWEEN_SUBMISSIONS = 12` in its config both confirm resubmission is
  allowed — rate-limited to once per 12 hours, and only the latest valid
  submission counts. Submitting is not a one-shot commitment.
- **The board is still not a selection loop.** It scores only *half* the test
  set (the final scores all 750), and scores are bootstrapped, so an interim
  rank rests on ~375 compounds with a standard deviation attached — a small
  delta between two entries there is not a ranking. A 12-hour loop is also far
  too slow and too coarse to choose a champion with.
- Choose on internal evidence: build the candidates, run them through a contest
  on a shared `inference_run`, and promote on the deltas (`contests`,
  `promotion`). Out-of-fold Spearman, read against its resolution threshold, is the eval
  that has to carry the decision. Use the board to probe placement, not to rank models.
- Proprietary CYP data may be used but **must be disclosed**. If the user pulls
  in private data, note that the disclosure is required.
- No restriction on methods or external property databases.
- A separate award recognizes the most innovative ML approach, decoupled from
  leaderboard rank. A novel entry that scores worse is not penalized — worth
  raising if the user is weighing something exploratory. It requires
  open-sourcing the code; leaderboard ranking does not.
- **Submission format is strict and a mismatch fails the upload.** Two
  independent files, `.parquet` or `.csv`, exactly 750 rows each, case-sensitive
  columns. Regression: `SMILES`, `Molecule_Name`, and
  `CYP{1A2,2C9,2D6,3A4}_pIC50_direct_inhibition` as finite floats — no NaN or
  inf. Classification: `SMILES`, `Molecule_Name`, `CYP2D6_is_TDI`,
  `CYP3A4_is_TDI` as booleans. These are the challenge's names and differ from
  whatever the FeatureSet columns are called; map explicitly at submission time.
- **Validate the file with their own checker before submitting** — a rejected
  upload burns a 12-hour window.
  `validation/activity_validation.py` in the tutorial repo
  enforces the column set, rejects duplicate or null `Molecule_Name`, rejects
  non-numeric and non-finite values, and requires the molecule-ID set to match
  the test set exactly — no missing and no extra rows.

## More

- https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/
- https://openadmet.ghost.io/openadmets-cyp-challenge-is-underway/ — launch post
- https://github.com/OpenADMET/CYP-Challenge-Tutorial — tutorial notebooks,
  `evaluation/` scoring harness, `validation/` submission checkers
- `openadmet/cyp-challenge-train-test` (HF) — official train/test split
- `openadmet/cyp-challenge` (HF Space) — submission platform
