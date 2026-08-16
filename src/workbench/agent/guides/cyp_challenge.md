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

Training data at launch: ~1,500 dose-response curves per CYP (sparse across
isoforms) plus primary screening on the Enamine DDS10 diversity set and an
FDA-approved set. Assays are biochemical with recombinant enzymes —
fluorescence for 3A4/2C9/1A2, acoustic-ejection mass spec for 2D6.

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

**How much forgiveness that actually buys is small.** On the Octant CYP3A4
curves the median interval is 0.094 pIC50 units wide (mean 0.250, pulled up by
the low-activity tail). So for well-measured compounds ST-RAE is close to plain
RAE, and the free pass only really applies out in the weak tail.

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

**Our `st_rae` is not comparable to a leaderboard number.** RAE needs a
denominator, and OpenADMET has never published a scoring script (the finished
PXR challenge's Space displays RAE but ships no metric code). We score the
mean-predictor baseline with the same soft threshold, so 1.0 keeps its "no
better than the mean" meaning; the plain `sum|y - ȳ|` denominator runs about 17%
lower on real data. The choice cannot change model *rankings* within an
endpoint — the denominator does not depend on the predictions — so it is safe
for selection, and only the absolute value is uncomparable.

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

## Model shape: multi-task across the four CYPs

Four correlated targets with sparse, unequal coverage is the textbook
multi-task case. `target_column` takes a list:

```python
model = fs.to_model(
    name="cyp-inhibition-chemprop-mt-reg",
    model_type=ModelType.UQ_REGRESSOR,
    target_column=["cyp3a4_pic50", "cyp2c9_pic50", "cyp2d6_pic50", "cyp1a2_pic50"],
    feature_list=features,
    hyperparameters={"uq_version": "v1"},
)
```

Missing targets are `NaN` per row — that is how sparsity is expressed; do not
drop rows to make the matrix dense. All four isoforms are end products here
(nothing is auxiliary), so unequal coverage is corrected with symmetric
weights:

```python
from workbench.utils.multi_task import compute_inverse_count_task_weights
```

Pass the result as `task_weights` in `hyperparameters`. Build the wide table
from per-isoform sources with `combine_multi_task_data` in the same module.

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

## Data already on hand

Two public sources, and they play different roles.

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

About 33,500 fitted curves across the four targets — roughly 5x the challenge's
own ~1,500-per-isoform training set. Three things to know before using it:

- **It is noisier than the challenge data.** Compounds assayed under multiple
  SIDs disagree by a median of 0.40 pIC50 units, against Octant's 0.094 median
  interval width. Treat it as pretraining signal with a down-weight, not as
  equal-weight training data.
- **The 42,355 "Inactive" rows are censored, not missing.** Those compounds were
  tested and showed no inhibition up to 57 uM — a real measurement saying
  pIC50 < ~4.2, with `pic50` NaN and `curve_class` 4. Dropping them discards two
  thirds of the screen, and it is exactly the low-activity regime ST-RAE and the
  TDI labeling rules are built around.
- **CYP2C19 is a free auxiliary task** — 9,544 curves of a correlated fifth
  isoform, not scored by the challenge.

Filter on `curve_class` (-1.1/-1.2 are complete curves) and `fit_r2` rather than
treating a single-point extrapolation like a full 15-point fit. Use `smiles`
(standardized); `smiles_orig` is the deposited string.

`pub_data.describe(...)` gives per-column meanings for any of these.

**The challenge's own train/test files are published by OpenADMET at launch and
have to be pulled in before they are available here.** Check `pub_data.list()`
for the path rather than assuming one — never invent a
`comp_chem/openadmet/cyp/...` path, and say plainly when it isn't there.
`data/public_data/pull_openadmet_data.py` is where a `cyp` entry goes, pointing
at the `openadmet/cyp-challenge-train-test` HuggingFace repo.

## Submission discipline

- **One account per team or lab** — not one submission. Submissions are
  rate-limited to one per 12 hours and the latest valid one counts, so there is
  a live leaderboard to iterate against. Whether the live board scores the full
  750 or a subset is unconfirmed; ask on Discord rather than assuming.
- A leaderboard loop at 12-hour granularity is far too slow and too coarse to
  select a champion. Choose on internal evidence — build the candidates, run
  them through a contest on a shared `inference_run`, and promote on the deltas
  (`contests`, `promotion`) — and use the board to check for calibration
  surprises, not to rank.
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

## More

- https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/
