# Data Cleanup

> find the data issues that cap model performance — censored targets, duplicates, activity cliffs

Read this when the user asks why a model underperforms, whether their data is any good, or
how to clean a dataset before training. None of these issues show up in a
predicted-vs-actual scatter plot.

| step | finds |
|---|---|
| `tag_molecules` + `filter_by_tags` | records fingerprints can't represent |
| `target_health(df, target)` | censored / discretized / skewed target column |
| `land.duplicates()` | same structure, different answer |
| `land.cliffs()` | steep target change between distinct neighbors |
| `land.isolated()` | chemistry with no support in the set |

The order matters for two mechanical reasons. Tagging first, because unrepresentable
records collapse onto each other and flood the duplicate report. Duplicates before cliffs,
because coincident rows divide by a distance of zero, so their unbounded score occupies
every slot in the cliff ranking.

## Before you start

Column names and units below are examples from public data. A private set names its
endpoints differently — read the columns rather than pattern-matching on a name you
recognize.

**Check whether one compound occupies several rows.** Internal datasets often carry a
compound id *and* a batch id, so a row is one measurement rather than one compound.

```python
df[id_column].duplicated().any()       # is the artifact's id per-compound or per-batch?
```

- **Id repeats** (id is the compound): roll up first. Proximity needs unique ids —
  `neighbors()` returns a garbled cartesian result when one id maps to several rows.
- **Id unique, compound repeats** (id is the batch): everything runs, but each compound's
  replicates form a coincident group, so `duplicates()` is measuring assay reproducibility
  rather than finding defects.

```python
from workbench.utils.chem_utils.misc import rollup_experimental_data

rolled = rollup_experimental_data(df, id="compound_id", time="assay_date", target="logd")
```

`use_gmean=True` for log-distributed endpoints. Measure the replicate spread before you
roll it away — it's the assay's noise floor and therefore the right `min_spread` later.
Failing that, ~2× a reported std-error column; failing both, `range/20` is a defensible
default to state out loud.

## 0. Drop what can't be represented

```python
from workbench.utils.chem_utils.mol_tagging import tag_molecules, filter_by_tags, get_tag_summary

tagged = tag_molecules(df)
get_tag_summary(tagged).filter(like="curation:")          # look before dropping
clean = filter_by_tags(tagged, exclude_prefix=["curation:exclude:"])
```

`curation:exclude:*` covers `inorganic`, `organometallic`, `mixture`, `mw_too_low` (lone
atoms) and `mw_too_high`. Fingerprints keep only the largest fragment, so a salt or
multi-component record hashes as its shared anion and the counterion — often the thing
that determines the property — is discarded entirely. On aggregated public data this can
be a sixth of the rows; on a curated pharma library, zero. See `cheminformatics`.

Say what you dropped and why. It's the user's call, not a silent preprocessing step.

## Getting a landscape

```python
fs = FeatureSet("open_admet_logd")
land = TargetLandscape(fs.prox("fingerprint", target="logd"))
```

For a plain DataFrame — a `DataSource`, or the filtered frame from step 0:

```python
from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
from workbench.algorithms.dataframe.target_landscape import TargetLandscape

land = TargetLandscape(FingerprintProximity(clean, id_column="id", target="logd"))
```

Fingerprints are the default for cleanup — structural identity is what makes a duplicate a
duplicate. Use `fs.prox("features", feature_list=[...])` when the question is about the
descriptor space a model actually trains on.

## 1. Target health

```python
from workbench.utils.pandas_utils import target_health
target_health(df, target)
```

One row per check, `severity` of `ok` / `info` / `warn`:

| check | what a warn means |
|---|---|
| `missing` | rows with no target value |
| `discretization` | few distinct values — the reporting grid floors achievable RMSE |
| `censoring` | rows stacked on the exact min or max — "> 300" recorded as 300 |
| `pileup` | one value carries an outsized share of rows |
| `skew` | \|skew\| > 2 |

`censoring` fires only on a *pileup* at the boundary, so an assay clipped at limits few
compounds reach still reports `ok` — read it alongside `discretization` and `range`. A
qualifier column in the raw data (`>`, `<`, `ND`) beats any inference from the numbers.

On a multi-task FeatureSet pass only the rows carrying the target, or `missing` just
reports the shape of the blend: `target_health(df[df[target].notna()], target)`.

## 2. Duplicates

```python
land.duplicates()                    # every coincident group
land.duplicates(min_spread=0.5)      # only groups disagreeing by at least 0.5
```

One row per group member, sorted by `group_spread` descending: `group_id`, `group_size`
(often 3 or 4, not just pairs), `group_spread` (max − min), `group_median`.

**Check the structures before calling a wide group an error.** If the members' `smiles`
differ it's a fingerprint collision, not a duplicate record, and collapsing it destroys a
real distinction. On a diverse library this is the common case rather than the exception.
`diff_molecules(a, b)` highlights everything outside the pair's common core, plus any
stereocenter or double bond whose configuration differs — see `plotting`.

What Morgan fingerprints don't encode:

- **Chirality** — `[C@H](O)` and `C(O)` collide: a racemate and its enantiomer, or one
  record annotated more completely. An `undefined_chiral_centers` column split across the
  members names this instantly.
- **Double-bond geometry** — `/C=C/` and `C=C` collide too, and `undefined_chiral_centers`
  will *not* flag it, since E/Z isn't a chiral center.
- **Salt forms and tautomers**, which standardize together upstream.

**Then check the names.** Same name with different values is a genuine curation error — the
one to actually fix. Different names where one is a salt or mixture (`niclosamide
ethanolamine salt` vs `niclosamide`) means the name says salt while the SMILES was recorded
as the pure parent; no structural check catches that.

**How much disagreement is normal depends on the assay.** Across ADMET endpoints the
duplicate rate is fairly stable while the fraction that actually disagree varies several
fold — physicochemical measurements like logD reproduce, cell and microsome assays don't.
Aggregating to the median is defensible on the former and averages away real assay variance
on the latter. Read a high wide-group fraction as a property of the assay before calling it
bad curation, and use the dataset's own reported error as the bar when it has one.

A wide spread between genuine stereoisomers may be real biology; between two records of the
*same* structure it's a defect. Only the second is safe to aggregate.

## 3. Cliffs

```python
land.cliffs(top_percent=1.0)
```

Coincident rows are already excluded. Ranked by `cliff_score` — target change normalized by
the target's range, divided by distance — so a threshold carries across endpoints with
different units.

**Every cliff appears as two rows**, once from each side of the pair, and the asymmetry is
the point. `nn_target_diff` is identical for both; `neighbor_median_diff` is not. It
compares each compound against the median of its *k* nearest neighbors, so the member with
the larger value is the one disagreeing with its whole neighborhood. When both are similar,
the neighborhood itself straddles a real SAR discontinuity.

Cliffs are a review queue, not a delete list.

## 4. Isolated

```python
land.isolated(top_percent=1.0)
```

Lowest nearest-neighbor similarity — coverage gaps rather than errors.
`land.proximity_stats()` shows the distribution behind it: if the 90th percentile of
`nn_similarity` is 1.0, a tenth of the set has a structural twin and the duplicate step
matters far more than the cliff step.

## Working the findings with the user

The steps above produce a review queue. The conversation after it is the useful part, and
it's collaborative — the user knows their chemistry and their assay, and you don't.

**Drill into anything they point at.** A `group_id`, a compound, a cliff pair — pull the
members, show the structures, show the neighborhood. `land.prox.neighbors(cmpd_id)` gives
the local context, `diff_molecules` shows what separates a pair, and the `proximity` and
`plotting` guides cover the visual tools.

**Carry their constraints forward.** "Ignore the salts", "only the kinase series", "we
don't trust that assay before March" — scope every subsequent step to it rather than
re-running the default pipeline and re-reporting what they just dismissed.

**Their judgment beats the defaults.** `curation:exclude:*` is a starting policy for
general ADMET modeling, not a verdict; `mol_tagging`'s own documentation says endpoint
specifics may want a different policy. If the user says an enantiomer pair is real, or that
a censored value is usable for their purpose, that settles it — don't re-litigate a call
they've made.

**Keep defects and findings separate** when you report. Censored targets and duplicate
conflicts are things to fix; cliffs and isolated compounds are things to look at. Counts
mean little without the denominator.
