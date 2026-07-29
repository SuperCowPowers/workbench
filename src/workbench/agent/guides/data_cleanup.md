# Data Cleanup

> find the data issues that cap model performance — censored targets, duplicates, activity cliffs

Read this when the user asks why a model underperforms, whether their data is any good,
or how to clean a dataset before training. Most ADMET datasets arrive with several of
these, and none of them show up in a predicted-vs-actual scatter plot.

Work in this order — each step changes what the next one sees:

| step | finds | cost |
|---|---|---|
| `tag_molecules` + `filter_by_tags` | records fingerprints can't represent | one RDKit pass |
| `target_health(df, target)` | censored / discretized / skewed target column | instant |
| `land.duplicates()` | same structure, different answer | one NN pass |
| `land.cliffs()` | steep target change between distinct neighbors | free after the above |
| `land.isolated()` | chemistry with no support in the set | free after the above |

Both orderings matter. Tagging is first because unrepresentable records collapse onto
each other and flood the duplicate report with findings no labeling fix can touch.
Duplicates precede cliffs because coincident rows divide by a distance of zero, so their
unbounded score occupies every slot in the cliff ranking.

## Before you start: look at the dataset

Every column name and unit below is an example from a public dataset. A private ADMET set
names its endpoints differently and may use different units, so read the columns and ask
rather than pattern-matching on a name you recognize. `cliff_score` is range-normalized and
survives any of that; the log-transform and censoring advice does not.

**Check whether one compound occupies several rows** — an internal dataset usually has a
compound id *and* a batch id, so one row is one measurement, not one compound:

```python
df[id_column].duplicated().any()       # is the artifact's id per-compound or per-batch?
```

- **Id repeats** (id is the compound): roll up first. Proximity needs unique ids —
  `neighbors()` returns a garbled cartesian result when one id maps to several rows.
- **Id unique, compound repeats** (id is the batch): everything runs, but each compound's
  replicates form a coincident group, so `duplicates()` is measuring assay reproducibility.
  Don't report that as bad data.

```python
from workbench.utils.chem_utils.misc import rollup_experimental_data

rolled = rollup_experimental_data(df, id="compound_id", time="assay_date", target="logd")
```

`use_gmean=True` for anything log-distributed (IC50, solubility, clearance).

Before rolling up, measure the replicate spread — it's the assay's own noise floor and
therefore the right `min_spread` later, since a group disagreeing by less than the assay
disagrees with itself isn't a finding. With no replicates, ~2× a reported std-error column
does the same job; absent both, `range/20` is a reasonable default to state out loud.

```python
df.groupby(compound_id)[target].agg(lambda x: x.max() - x.min()).describe()
```

## 0. Drop what can't be represented

```python
from workbench.utils.chem_utils.mol_tagging import tag_molecules, filter_by_tags, get_tag_summary

tagged = tag_molecules(df)
get_tag_summary(tagged).filter(like="curation:")          # look before dropping
clean = filter_by_tags(tagged, exclude_prefix=["curation:exclude:"])
```

`curation:exclude:*` covers `inorganic`, `organometallic`, `mixture`, `mw_too_low` (lone
atoms) and `mw_too_high`. On AqSol this drops ~17% of rows and over half the coincident
groups — sulfate and carbonate salts whose counterion the fingerprint discards, not label
noise. The `cheminformatics` guide explains the mechanism.

Report what you dropped and why. Excluding a sixth of a dataset is the user's decision to
see, not a silent preprocessing step.

## Getting a landscape

```python
fs = FeatureSet("open_admet_logd")
land = TargetLandscape(fs.prox("fingerprint", target="logd"))
```

For a plain DataFrame — a `DataSource`, or the filtered frame from step 0 — build the
backend directly:

```python
from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
from workbench.algorithms.dataframe.target_landscape import TargetLandscape

land = TargetLandscape(FingerprintProximity(clean, id_column="id", target="logd"))
```

Fingerprints are the right default for cleanup — structural identity is what makes a
duplicate a duplicate. Use `fs.prox("features", feature_list=[...])` when the question is
about the descriptor space a model actually trains on.

## 1. Target health

```python
from workbench.utils.pandas_utils import target_health
target_health(df, "hlm_clint")
```

One row per check with a `severity` of `ok` / `info` / `warn`:

| check | what a warn means |
|---|---|
| `missing` | rows with no target value |
| `discretization` | few distinct values — the assay's reporting grid floors achievable RMSE |
| `censoring` | rows stacked on the exact min or max — "> 300" recorded as 300 |
| `pileup` | one value carries an outsized share of rows |
| `skew` | \|skew\| > 2 — a log transform is usually the fix |

Censored rows are bounds, not measurements. Training on them teaches the model that a
whole family of compounds has exactly the boundary value. Drop them, or model the
endpoint as bounded.

Two limits worth knowing. `censoring` fires only on a *pileup* at the boundary, so an
assay clipped at limits few compounds reach reports `ok` — read it alongside
`discretization` and the reported `range`. Internal datasets censor more than public ones,
since "> 100" and "< 1" are how an assay reports out-of-range results; a qualifier column
in the raw data is better evidence than any inference from the numbers.

On a **multi-task** FeatureSet, pass only the rows carrying the target, or `missing`
reports the shape of the blend:

```python
target_health(df[df["pec50"].notna()], "pec50")
```

## 2. Duplicates

```python
land.duplicates()                    # every coincident group
land.duplicates(min_spread=0.5)      # only groups disagreeing by at least 0.5
```

One row per group member, sorted by `group_spread` descending:

| column | meaning |
|---|---|
| `group_id` | the coincident group |
| `group_size` | members in it — frequently 3 or 4, not just pairs |
| `group_spread` | max − min of the target within the group |
| `group_median` | the group's consensus value |

Near-zero spread means redundant rather than contradictory — collapse and move on. Wide
groups are the find: the same input mapped to two answers, so whatever the model learns
there is noise.

**How much disagreement is normal depends on the assay.** Across the OpenADMET endpoints
the duplicate *rate* is nearly constant (~13%) while the fraction that actually disagree
runs from 12% on logD to 44% on microsomal clearance and plasma-protein binding —
physicochemical measurements reproduce, cell and microsome assays don't. So aggregating is
defensible on a logD-like endpoint and averages away real variance on a clearance-like one.
Read a high wide-group fraction as a property of the assay before calling it bad curation.

**Check the structures before calling a wide group an error.** If the members' `smiles`
differ, it's a fingerprint collision rather than a duplicate record, and collapsing it
destroys a real distinction. On a diverse library this is the common case — every
coincident group in the PXR training set is a stereochemistry pair, so median-aggregating
them would be wrong. `diff_molecules(a, b)` highlights everything outside the pair's
common core, so collisions light up and stereo-only pairs stay blank (see `plotting`).

What Morgan fingerprints don't encode, in the order it bites:

- **Chirality** — `[C@H](O)` and `C(O)` collide: a racemate and its enantiomer, or one
  record annotated more completely. An `undefined_chiral_centers` column split `0`/`1`
  across the members names this instantly.
- **Double-bond geometry** — `/C=C/` and `C=C` collide too, and
  `undefined_chiral_centers` will *not* flag it. Read the SMILES.
- **Salt forms and tautomers**, which standardize together upstream.

**Then check the names.** A group that survives step 0 and still disagrees is one of two
things:

- **Same name, different value** — a genuine curation error, and the one to fix. AqSol
  has only 7 (propoxyphene, reserpine, simvastatin, rotenone, naproxen, nandrolone),
  1–2.4 log units apart with both records reporting `sd = 0`.
- **Different names, one a salt or mixture** — `niclosamide ethanolamine salt` vs
  `niclosamide`. The name says salt, the SMILES was recorded as the pure parent, so no
  structural check catches it. A record defect, not a measurement conflict.

Where the dataset reports its own experimental error, use it: 130 of AqSol's 141 wide
groups disagree by more than 2× their own reported `sd`, which rules out replicate noise.

A wide spread between genuine stereoisomers may be real biology. A wide spread between
two records of the *same* structure is a defect. Only the second is safe to aggregate.

## 3. Cliffs

```python
land.cliffs(top_percent=1.0)
```

Coincident rows are already excluded. Ranked by `cliff_score` — target change normalized
by the target's range, divided by distance — so a threshold means the same thing on a
pIC50 endpoint as on clearance in µL/min/mg.

**Every cliff appears as two rows**, once from each side of the pair, and the asymmetry
is the point. `nn_target_diff` is identical for both; `neighbor_median_diff` is not. It
compares each compound against the median of its *k* nearest neighbors, so the member
with the larger value is the one disagreeing with its whole neighborhood — the suspect.
When both are similar, the neighborhood itself straddles a real SAR discontinuity.

A cliff is not automatically an error; a real activity cliff is the most interesting
chemistry in the set. Treat the ranking as a review queue, not a delete list.

## 4. Isolated

```python
land.isolated(top_percent=1.0)
```

Lowest nearest-neighbor similarity — compounds with no structural support in the set.
Not errors but coverage gaps: a model will predict them badly and be right to be
uncertain, so they belong in a holdout or the next round of data collection.

`land.proximity_stats()` shows the distribution behind this. If the 90th percentile of
`nn_similarity` is 1.0, more than a tenth of the set has a structural twin and the
duplicate step matters far more than the cliff step.

## Reporting back

Lead with counts and percentages against set size — "308 duplicate groups" means
something different in 5,000 rows than in 500. Name the specific compounds for anything
the user has to adjudicate, and keep the distinction sharp: censored targets and
duplicate conflicts are defects to fix; cliffs and isolated compounds are findings to
review.
