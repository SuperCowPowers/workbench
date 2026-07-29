# Data Cleanup

> find the data issues that cap model performance — censored targets, duplicates, activity cliffs

Read this when the user asks why a model underperforms, whether their data is any
good, or how to clean a dataset before training. Most ADMET datasets arrive with all
three problems, and none of them show up in a predicted-vs-actual scatter plot.

Work in this order — each step changes what the next one sees:

| step | finds | cost |
|---|---|---|
| `tag_molecules` + `filter_by_tags` | records fingerprints can't represent | one RDKit pass |
| `target_health(df, target)` | censored / discretized / skewed target column | instant |
| `land.duplicates()` | same structure, different answer | one NN pass |
| `land.cliffs()` | steep target change between distinct neighbors | one NN pass |
| `land.isolated()` | chemistry with no support in the set | free after the above |

Two of these orderings matter. **Tagging comes first** because inorganics, mixtures and
lone atoms collapse onto each other in fingerprint space, and left in they dominate the
duplicate report with findings no labeling fix can touch. **Duplicates come before
cliffs** because coincident rows divide by a distance of zero, so their score is
unbounded — they occupy every slot in the cliff ranking and hide the real ones.

## 0. Drop what can't be represented

```python
from workbench.utils.chem_utils.mol_tagging import tag_molecules, filter_by_tags, get_tag_summary

tagged = tag_molecules(df)
get_tag_summary(tagged).filter(like="curation:")          # see what's there first
clean = filter_by_tags(tagged, exclude_prefix=["curation:exclude:"])
```

`curation:exclude:*` covers `inorganic`, `organometallic`, `mixture`, `mw_too_low`
(lone atoms), and `mw_too_high`. On AqSol this drops ~17% of rows and removes more than
half of the coincident groups — those groups were sulfate and carbonate salts whose
counterion the fingerprint discards, not label noise. See the `cheminformatics` guide
for why.

Report what you dropped and why. Excluding a sixth of a dataset is a decision the user
should see, not a silent preprocessing step.

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

On a **multi-task** FeatureSet, pass only the rows that carry the target — otherwise
`missing` reports the shape of the blend rather than a defect:

```python
target_health(df[df["pec50"].notna()], "pec50")
```

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

**Check the structures before calling a wide group an error.** Diff the members' `smiles`
first: if they differ, the group is a fingerprint collision rather than a duplicate
record, and collapsing it destroys a real distinction. On a diverse screening library
this is the common case, not the exception — every coincident group in the PXR training
set is a stereochemistry pair, so median-aggregating them would be wrong.

What Morgan fingerprints don't encode, in the order it bites:

- **Chirality.** `[C@H](O)` and `C(O)` are coincident — a racemate and its single
  enantiomer, or the same compound annotated to different completeness. Where the
  FeatureSet carries an `undefined_chiral_centers` column, a `0` / `1` split across the
  group's members identifies this immediately.
- **Double-bond geometry.** `/C=C/` and `C=C` are coincident too, and
  `undefined_chiral_centers` will *not* flag it — E/Z isn't a chiral center. Read the
  SMILES for this one.
- **Salt forms and tautomers**, which standardize to the same structure upstream.

A wide spread between two genuine stereoisomers may be real biology. A wide spread
between two records of the *same* structure is a defect. Only the second is safe to
aggregate.

To see the difference instead of reading SMILES by eye, `diff_molecules(a, b)` highlights
everything outside the pair's common core — extra counterions and fragments light up,
while a stereo-only pair highlights nothing. See the `plotting` guide.

**Then check the names.** A group that survives step 0 and still disagrees is one of two
things, and a `name` column tells them apart at a glance:

- **Same name, different value** — a genuine curation error. On AqSol only 7 groups look
  like this (propoxyphene, reserpine, simvastatin, rotenone, naproxen, nandrolone at
  1–2.4 log units apart, both records reporting `sd = 0`). These are the ones to fix.
- **Different names, one of them a salt or mixture** — `Terbinafine hydrochloride` vs
  the free base, `niclosamide ethanolamine salt` vs `niclosamide`. The name says salt
  but the SMILES was recorded as the pure parent, so no structural check catches it.
  Treat as a record defect, not a measurement conflict.

Where the dataset reports its own experimental error, use it. AqSol carries `sd` and
`ocurrences` per record; 130 of its 141 wide groups disagree by more than 2× their own
reported `sd`, which rules out replicate noise as the explanation and points at the
structures instead.

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
