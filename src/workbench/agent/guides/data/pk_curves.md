# PK Curve Data Quality

> find what is wrong with measured concentration-time curves and the NCA parameters derived from them, before they become model targets

Read this when the user hands you measured PK data — timepoints, study summaries, or the
parameter tables behind a CL, Vss or F model — and asks whether it is any good. For
plausibility checks on *predicted* PK parameters see `pk_data`; for the PDF itself see
`reports`.

Column names below are the defaults of the functions used. A real table names things
differently — read the columns rather than pattern-matching.

## The three tables

| table | one row per | typical home |
|---|---|---|
| timepoints | sample: curve, time, concentration, BLQ flag | `DFStore` |
| study summaries | study arm: Cmax, Tmax, AUC, t½, CL, Vss, F as reported | `DataSource` |
| model targets | compound: one aggregated parameter | `FeatureSet` |

```python
curves = DFStore().get("/path/to/conc_time")
reported = DataSource("pk_study_summaries").pull_dataframe()
```

Each level hides defects the next one inherits, so work top to bottom. Check what a
curve is before anything else: with no animal id column it is a group mean, and
animal-to-animal variability is not checkable. Say so in the report.

## 1. Scope

Decide what is in the analysis and state what was set aside, with counts.

- **Not the parent drug in plasma.** Metabolite, unbound, probe-substrate and tissue
  (brain, liver, CSF) curves answer different questions. Separate them; do not pool.
- **Units.** One concentration unit per analysis. A handful of rows in another unit is a
  defect to list, not something to convert silently.
- **Matrix and route.** Blood and plasma differ by the blood-to-plasma ratio. IV and
  extravascular curves are analysed differently and `nca()` needs the route to know which.
- **Molecular weight** when concentrations are molar and reported values are mass-based:

```python
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

mol_wt = smiles.map(lambda s: MolWt(Chem.MolFromSmiles(s)))    # ng/mL = nM * MW / 1000
```

## 2. Sample-level checks

Plain pandas. These are quick, and the surprising ones are specific to the dataset, so
look at `value_counts()` on every low-cardinality column rather than running a fixed list.

| check | a hit means |
|---|---|
| category spellings | a find-and-replace or parsing accident upstream |
| missing dose, compound id, study id | the curve cannot be dose-normalized or joined |
| how BLQ is stored | `conc == 0` with a flag is fine; a zero with no flag, or an LLOQ value with no flag, is not |
| duplicate (curve, time) | two curves merged under one id |
| negative or zero time on an extravascular curve | pre-dose sample, or a time-unit mix-up |
| provenance flags | where the table carries extraction flags (ambiguous dose, inferred matrix, conflicting file revisions), count curves per flag |

## 3. Curve-level checks

```python
from workbench.utils.pk_utils import nca

summary = nca(curves)        # one row per curve, indexed by curve id
```

`nca()` uses the Phoenix WinNonlin conventions and returns NaN for anything it cannot
compute. It does **not** judge the terminal fit: a flat or rising tail still produces a
`lambda_z`, with a poor `lambda_z_r2` and a large `extrap_percent`. Everything downstream
of the terminal slope — `auc_inf`, `half_life`, `mrt`, and CL, Vss, Vz, F — is only as
good as that fit, so gate before using them.

| check | a hit means |
|---|---|
| `lambda_z` is NaN | fewer than 3 samples after Cmax, or no declining fit: no terminal phase |
| `lambda_z_r2` low (adjusted R², commonly < 0.8) | the tail is not log-linear; t½ is not determined |
| `extrap_percent` high (commonly > 20) | most of AUCinf is extrapolated; sampling stopped too early |
| `tlast` short against `half_life` | same cause, seen from the other side |
| `tmax` at the first sample, extravascular | the absorption phase was missed; Cmax is a lower bound |
| `tmax` at the last sample | still rising at the end; nothing terminal was observed |
| a second peak hours after the first | repeat dosing or enterohepatic recirculation in a single-dose table |
| `n_blq` a large share of `n_points` | the curve is mostly below quantitation |

The thresholds are conventions, not rules. State the ones you used and show how the
counts move if they are loosened.

Look at the curves you flag — the plot settles most questions faster than a statistic:

```python
from workbench.utils.plots import pk

worst = summary.sort_values("lambda_z_r2").index[:12]
fig = pk.nca_curves(curves, worst, summary=summary)
```

## 4. Cross-level checks

Recompute the reported parameters from the curves and compare. This is the strongest
check available, because neither table can reveal the disagreement alone.

```python
from workbench.utils.pk_utils import nca_parameters

arms = curves.groupby("curve_id")[["compound_id", "route", "dose_mg_per_kg", "matrix"]].first()
arms["mol_wt"] = arms["compound_id"].map(mol_wt)
params = nca_parameters(summary, arms, mol_wt="mol_wt")
```

`params` carries `cmax_ng_ml`, `auc_last_ng_h_ml`, `auc_inf_ng_h_ml`, and for IV curves
`cl_ml_min_kg`, `vss_l_kg`, `vz_l_kg`; other routes get `f_percent` against the
compound's IV curves. Tighten the IV reference with `pair_on=["compound_id", "matrix"]`
when a compound was run in more than one matrix or study.

Join to the study summaries on batch, route, dose and matrix, then compare as a ratio:

| parameter | expected agreement |
|---|---|
| Cmax, AUClast | tight — these are read almost directly off the samples |
| AUCinf, CL, Vss | tight where the terminal fit is good |
| t½, Tmax | looser — a reported value is often the mean of per-animal fits, not a fit to the mean curve |

- **The join is a finding.** Count curves with no reported row, reported rows with no
  curve, and curves matching several rows. Report them; do not force a match.
- **The tails are the defects.** A median ratio near 1 validates the method. The arms
  off by more than ~20% are where a unit, dose, or transcription error lives.
- **A consistent offset is a definition.** A parameter that is systematically high or low
  across every arm is a column that means something other than its name — check it
  against the other columns (`CL * t½ / ln 2` for Vz, `CL * MRT` for Vss).

## 5. Parameter-level checks

On the measured parameters, one row per arm or per compound:

- `plausibility_flags` (see `pk_data`) for impossible values and the F-against-CL ceiling.
- `target_health` per parameter, on the log scale, and the duplicate and cliff steps in
  `data_cleanup` once the table is one row per compound.
- **Dose linearity.** Dose-normalized AUC across arms of the same compound and route
  should be flat. A trend is saturation, not noise.
- **F is per dose level.** A common design is one low IV dose and several escalating PO
  doses. Clearance or first-pass saturates at the high PO doses, dose-normalized AUC
  rises, and F drifts past 100%. That is nonlinear PK, not measurement error.
    - Report F per PO dose and plot it against the PO/IV dose ratio. A rising trend is
      saturation.
    - F above 100% at a high dose ratio is a review item; at a ratio near 1 it is a
      likely defect.
    - Check which arm the model target uses. The lowest PO dose is closest to the linear
      range. Confirm it from the data rather than assuming.
    - A target clipped to 100 reads as fully bioavailable when it means saturated at
      that dose. Count the clipped rows.
- **Arm-to-arm spread.** Arms of one compound under the same conditions that disagree
  are the assay's reproducibility, measured.

## 6. What it means for a model

Close the report with this, since the reader's next step is training.

- **Quality is a filter or a weight.** `lambda_z_r2` and `extrap_percent` are per-curve
  confidence in a CL, Vss, Vz or t½ target. Show how many targets survive each gate.
- **Disagreement is the noise floor.** The spread between recomputed and reported values,
  and between arms of one compound, is error no model can beat. Quote it in the target's
  own units, on the log scale.
- **Vss and Vz are not interchangeable targets.** Vss = CL · MRT is the steady-state
  volume and the usual modeling target. Vz = CL / lambda_z leans entirely on the terminal
  slope, so it inherits every weakness of that fit and needs the tighter gate. A column
  labelled only "Vd" is one or the other: identify which before training on it.
- **Pooled conditions are a choice.** Where matrix, strain, sex or dose level are mixed
  into one target, show how much each one moves the parameter.

## Reporting

- **Defects and review items are separate lists.** A wrong unit or a mislabeled curve is
  something to fix. A poorly determined terminal phase is something to decide about.
- **Every count has a denominator**, at the right grain: samples, curves, arms or compounds.
- **One exemplar figure per finding**, from `pk.nca_curves`.
- **Say what could not be checked**, and why.
- The user knows the studies. A flagged curve is a question for them, not a verdict.
