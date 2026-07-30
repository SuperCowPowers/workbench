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

**Dates:** launches 2026-08-17, intermediate submission 2026-09-24, final
submission 2026-11-03. Half the test set is on a live leaderboard (split by
chemical series); half stays blind until the end.

## Scoring drives the modeling

**ST-RAE gives zero error inside the ground-truth credible interval.** Each
pIC50 label carries a Bayesian credible interval from the curve fit, and a
prediction landing inside it is scored as perfect. Intervals widen at low
activity, and compounds below pIC50 4 (under the lowest tested concentration)
are downweighted further.

What follows from that:

- **Accuracy on potent, tightly-measured compounds is where the score lives.**
  The weak/inactive tail is cheap to miss. Don't spend the modeling budget
  flattening errors on compounds that are downweighted anyway.
- It is *relative* absolute error, so this is not a license to regress
  everything toward the mean — that surrenders the potent end.
- Report the training labels' own intervals when discussing accuracy. A model
  whose residuals sit inside assay noise is already scoring zero there, and
  chasing it further is wasted effort.
- Our `confidence` and `q_05`/`q_95` outputs are **not** submitted and do not
  score. They are for deciding where to hedge, not part of the entry (`uq`).

MCC on the TDI track is chosen for imbalanced labels — accuracy will look good
and mean nothing. Quote MCC, and check the positive-class rate before claiming
a classifier works.

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
- **Cross-validation will overstate performance.** Use scaffold or Butina
  splits (`hyperparameters={"split_strategy": "butina"}`) and say plainly that
  CV numbers are optimistic for this test set.
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

## Data already on hand

Related CYP3A4 dose-response from the same OpenADMET/Octant source is in public
data, with credible-interval columns in the same shape the challenge scores on:

```python
inhibition_df = pub_data.get("comp_chem/openadmet/octant_cyp/inhibition")   # 1340 rows, CYP3A4 pIC50 + CI
reactivity_df = pub_data.get("comp_chem/openadmet/octant_cyp/reactivity")   # 2446 rows, multi-enzyme turnover
```

The inhibition table carries QC columns (`drc_qc_status`, `activity_status`,
`rollover_status`, `saturation_status`) — filter to clean curves before
training. `pub_data.describe(...)` gives the per-column meanings.

**The challenge's own train/test files are not in public data.** They do not
exist until the 2026-08-17 launch. Never invent a
`comp_chem/openadmet/cyp/...` path; check `pub_data.list()` and say it isn't
there yet.

## Submission discipline

- **One submission per team**, and no mid-challenge test-set release. There is
  no iterate-against-the-leaderboard loop, so the champion has to be chosen on
  internal evidence — build the candidates, run them through a contest on a
  shared `inference_run`, and promote on the deltas (`contests`, `promotion`).
- Proprietary CYP data may be used but **must be disclosed**. If the user pulls
  in private data, note that the disclosure is required.
- No restriction on methods or external property databases.
- A separate award recognizes the most innovative ML approach, decoupled from
  leaderboard rank. A novel entry that scores worse is not penalized — worth
  raising if the user is weighing something exploratory (3D conformer/xTB
  features have a plausible mechanism here, since CYP inhibition is
  pocket-binding driven).

## More

- https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/
