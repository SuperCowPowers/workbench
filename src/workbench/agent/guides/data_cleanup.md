# Data Cleanup

> find the data issues that cap model performance — censored targets, duplicates, activity cliffs

Read this when the user asks why a model underperforms, whether their data is any
good, or how to clean a dataset before training. Most ADMET datasets arrive with all
three problems, and none of them show up in a predicted-vs-actual scatter plot.

Work in this order — each step changes what the next one sees:

| step | finds | cost |
|---|---|---|
| `target_health(df, target)` | censored / discretized / skewed target column | instant |
| `land.duplicates()` | same structure, different answer | one NN pass |
| `land.cliffs()` | steep target change between distinct neighbors | one NN pass |
| `land.isolated()` | chemistry with no support in the set | free after the above |

Duplicates before cliffs is not a preference. Coincident rows divide by a distance of
zero, so their score is unbounded — on a set with duplicates they occupy every slot in
the cliff ranking and hide the real ones.

## Getting a landscape

`FeatureSet` is the usual input; `DataSource` works through a DataFrame.

```python
fs = FeatureSet("open_admet_logd")
prox = fs.prox("fingerprint", target="logd")     # or "features" with a feature_list
land = TargetLandscape(prox)
```

```python
from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
from workbench.algorithms.dataframe.target_landscape import TargetLandscape

df = DataSource("open_admet").pull_dataframe()
land = TargetLandscape(FingerprintProximity(df, id_column="id", target="logd"))
```

A fingerprint backend is the right default for cleanup — structural identity is what
makes a duplicate a duplicate. Use `"features"` when the question is about the
descriptor space a model actually trains on.

## 1. Target health

```python
from workbench.utils.pandas_utils import target_health
target_health(df, "hlm_clint")
```

One row per check with a `severity` of `ok` / `info` / `warn`:

| check | what a warn means |
|---|---|
| `missing` | rows with no target value |
| `discretization` | few distinct values — the assay's reporting grid puts a floor under achievable RMSE |
| `censoring` | rows stacked on the exact min or max — "> 300" recorded as 300 |
| `pileup` | one value carries an outsized share of rows |
| `skew` | \|skew\| > 2 — a log transform is usually the fix |

Censored rows are not measurements, they're bounds. Training on them teaches the model
that a whole family of compounds has exactly the boundary value. Drop them, or model
the endpoint as bounded.

Note the limit of the `censoring` check: it fires on a *pileup* at the boundary. An
assay clipped at limits few compounds reach reports `ok` — read it alongside
`discretization` and the reported `range` rather than on its own.

## 2. Duplicates

```python
land.duplicates()                    # every coincident group
land.duplicates(min_spread=0.5)      # only groups that disagree by at least 0.5
```

One row per group member, sorted by `group_spread` descending:

| column | meaning |
|---|---|
| `group_id` | the coincident group |
| `group_size` | members in it — frequently 3 or 4, not just pairs |
| `group_spread` | max − min of the target within the group |
| `group_median` | the group's consensus value |

Groups with `group_spread` near zero are redundant rather than contradictory — collapse
them and move on. Wide groups are the real find: the same input mapped to two answers,
so whatever the model learns there is noise. Aggregate to the median, or drop the group
if the disagreement is too large to reconcile.

Two caveats before calling a wide group an error:

- Morgan fingerprints don't encode chirality, so **enantiomers are coincident**. A large
  spread between two stereoisomers may be real biology, not a bad record.
- Salt forms and tautomers standardize to the same structure. Check `smiles` before
  deleting anything.

## 3. Cliffs

```python
land.cliffs(top_percent=1.0)
```

Coincident rows are already excluded. Ranked by `cliff_score` — the target change
normalized by the target's own range, divided by distance — so a threshold means the
same thing on a pIC50 endpoint as on clearance in µL/min/mg.

**Every cliff appears as two rows**, once from each side of the pair. That's deliberate:
the pair is symmetric, but the compounds are not, and the asymmetry tells you which one
to suspect. `nn_target_diff` is identical for both rows; `neighbor_median_diff` is not.
It compares each compound against the median of its *k* nearest neighbors, so the
member with the larger `neighbor_median_diff` is the one disagreeing with its whole
neighborhood — the outlier. When both are similar, the neighborhood itself straddles a
genuine SAR discontinuity.

A cliff is not automatically an error. A real activity cliff is the most interesting
chemistry in the set. Treat the ranking as a review queue, not a delete list.

## 4. Isolated

```python
land.isolated(top_percent=1.0)
```

Lowest nearest-neighbor similarity — compounds with no structural support anywhere in
the set. They aren't data errors; they're coverage gaps. A model will predict them
badly and be right to be uncertain, so they belong in a holdout or in the next round of
data collection rather than in the trash.

`land.proximity_stats()` gives the distribution behind this — if the 90th percentile of
`nn_similarity` is 1.0, more than a tenth of the set has a structural twin and the
duplicate step matters far more than the cliff step.

## Reporting back

Lead with counts and percentages against set size, since "308 duplicate groups" means
something different in 5,000 rows than in 500. Name the specific compounds for anything
the user has to adjudicate, and keep the distinction sharp: censored targets and
duplicate conflicts are defects to fix, while cliffs and isolated compounds are findings
to review.
