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

**Workbench computes ST-RAE.** `compute_regression_metrics` adds an `st_rae`
column whenever the target carries `<target>_ci_lower`/`_ci_upper`, so it lands
in model metrics and the dashboard with no extra work:

```python
from workbench.utils.metrics_utils import macro_soft_threshold_rae, soft_threshold_rae
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

The denominator is the plain `sum|y - ȳ|`, not a soft-thresholded baseline. Our
`soft_threshold_rae` defaults to `soft_baseline=True`, which scores the
mean-predictor baseline through the same soft threshold so 1.0 keeps its "no
better than the mean" reading; that runs about 17% higher than the plain form.
**Pass `soft_baseline=False` when the goal is a number comparable to the
leaderboard**, and keep the default for internal model selection. The choice
cannot change model *rankings* within an endpoint — the denominator does not
depend on the predictions — so only the absolute value is affected.

**Leaderboard scores are bootstrapped**: 1,000 resamples at a fixed seed, with
the spread reported alongside each score. Combined with the live board scoring
only half the test set, a small gap between two entries is inside the noise —
read the spread before treating a rank as a result.

**Our analog holdout has a noise floor of about 0.03 macro ST-RAE**, measured by
bootstrapping the 529 held-out compounds. A delta smaller than that is not a
result. Two models are compared with a *paired* bootstrap, not by eyeballing each
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

## Leaderboard baselines — what the numbers actually look like

OpenADMET posted reference entries on the real blind set. These are the calibration
points; quote them before calling any internal score good or bad.

| Entry | MA-ST-RAE | MA-MAE | MA-R² | Spearman ρ |
|---|---|---|---|---|
| TabICL-baseline | **0.676** | 0.857 | 0.293 | 0.681 |
| CheMeleon-baseline | 0.834 | 0.971 | 0.094 | 0.647 |
| LGBM-baseline | 0.893 | 1.000 | 0.066 | 0.598 |
| XGB-baseline | 0.898 | 1.005 | 0.063 | 0.597 |

What to read from them:

- **0.676 is the bar.** A tabular in-context model (TabPFN family) beats a learned
  molecular embedding and both gradient-boosted trees, by a wide margin.
- **Descriptor GBMs cluster at 0.89-0.90.** That is the descriptor floor on the real
  test set, so an XGBoost reference landing near 0.9 is performing normally, not
  broken.
- **R² is near zero for everything except the leader** (0.05-0.09) **while Spearman
  runs 0.58-0.68.** Models rank compounds well and get absolute values wrong — the
  signature of predictions compressed toward the mean. That is where the headroom is,
  and it argues for calibration work over architecture work. Do not read a low R² on
  this challenge as a broken model.
- ST-RAE and MAE rank these entries identically, so the soft threshold is not
  reshuffling anything at this level of performance.

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
- **Evaluate on an analog holdout, not cross-validation.**
  `analog_holdout_split` reproduces how this test set was built — top hits per
  target plus each hit's nearest neighbors, held out together:

  ```python
  from workbench.training.splits import analog_holdout_split
  ```

  Measured on the public Veith CYP3A4 data: a random split of the *same size*
  makes a baseline look 2.1x more accurate (MAE 0.304) than the analog holdout
  does (MAE 0.637). Quote the analog number; a CV number is the optimistic one.
  Butina (`hyperparameters={"split_strategy": "butina"}`) is still the right
  *training* fold strategy — it answers "new chemotypes?", which is a different
  question from "new analogs of known hits?".
- The holdout runs small (178 of 9,377 rows at 25 hits x 10 analogs, since
  neighbors overlap between hits). Raise `n_hits` before trusting it to pick a
  champion.
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

Keep the full `_direct_inhibition` target names. `compute_regression_metrics`
finds a credible interval by appending `_ci_lower` to the target name, so
shortening the targets silently drops `st_rae` — the challenge's own metric.

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
- On PXR, 3D ranked high in SHAP and *still* failed to transfer to the analog
  set. The verdict comes from the analog holdout, never from CV or feature
  importance.

## Start from the built FeatureSets

Five FeatureSets are already built from the challenge data and onboarded. Use
them rather than rebuilding from `PublicData` — they carry the decisions below
(credible intervals present, TDI labels de-leaked, challenge target naming) and
rebuilding re-derives all of it, usually getting one wrong.

| FeatureSet | rows | cols | what it is |
|---|---|---|---|
| `openadmet_cyp_f1` | 4,905 | 423 | Regression track, 2D + **v1** 3D |
| `openadmet_cyp_f2` | 4,905 | 373 | Regression track, 2D + **v2** 3D (curated, xTB) |
| `openadmet_cyp_tdi_f1` | 6,145 | 409 | TDI track, 2D + v1 3D |
| `openadmet_cyp_tdi_f2` | 6,145 | 359 | TDI track, 2D + v2 3D |
| `openadmet_cyp_veith` | 14,432 | 11 | Veith pretraining, SMILES + 5 targets, no descriptors |

- **`f1` vs `f2` differ only in the 3D layer** — same rows, same 2D block, same
  labels. That makes them a controlled A/B for whether the xTB electronic block
  earns its place. Hold everything else constant when comparing them.
- **The regression FeatureSets carry `_ci_lower`, `_ci_upper` and `_std` per
  isoform.** That is what makes `st_rae` computable, and it is also the leakage
  trap below — they are label metadata, never features.
- **The TDI FeatureSets carry labels only**, no arm pIC50s: `is_tdi` is derived
  from the shift between the direct and TDI arms, so carrying either arm beside
  the label hands the model the answer. To re-derive or audit labels, go back to
  `PublicData`.
- **`openadmet_cyp_veith` has no descriptors and no credible intervals** — SMILES
  plus five targets, so it is chemprop-only and cannot produce an `st_rae`. Its
  censored inactives were dropped at build time, so CYP3A4 bottoms out at pIC50
  4.20 and CYP1A2 at 4.10 while CYP2D6 reaches 2.05. It teaches the potent end and
  says almost nothing about the low-activity regime ST-RAE is built around.
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
platform convention, and what `compute_regression_metrics` reads to emit `st_rae`.
Map back to the challenge's exact column names at submission time.

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
all_df = pub_data.get("comp_chem/pubchem/cyp_inhibition/all_isoforms")  # 85,535 rows, 17,107 compounds
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

Local models compute metrics the same way AWS ones do, so `st_rae` comes along for
free as long as the credible-interval columns reach the predictions:

```python
local_model.list_inference_runs()      # "full_cross_fold" plus any endpoint captures
local_model.get_inference_metrics()    # st_rae here means the CI columns survived
local_model.get_inference_predictions("full_cross_fold")
```

If `st_rae` is missing from that frame, the CI columns were dropped somewhere in
FeatureSet -> training -> capture; score directly with `macro_soft_threshold_rae`
over the predictions and fix the column plumbing rather than assuming the metric
does not apply. Whichever way, use the analog holdout as the eval set — a
cross-fold number is the optimistic one.

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
  `promotion`). The analog holdout is the eval that has to carry the decision.
  Use the board to catch calibration surprises, not to rank.
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
