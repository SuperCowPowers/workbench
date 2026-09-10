# Plotting

> readable plots: matplotlib, plus HPO, neighborhood, and PK concentration-time profiles

Use **matplotlib** for anything you build yourself. Make sure text, legends, and axis
labels have enough space to be readable — a cramped plot is a useless plot.

The bundled plots in `workbench.utils.plots` are module-qualified — `hpo.parallel_coordinates`,
`neighborhood.graph`, `pk.bateman`. Reach for one before hand-rolling the same chart.

## Readability first

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
ax.set_xlabel("Actual", fontsize=12)
ax.set_ylabel("Predicted", fontsize=12)
ax.set_title("pxr-reg-chemprop — predicted vs actual", fontsize=13)
ax.tick_params(labelsize=11)
plt.show()
```

Rules that keep plots legible:

- `figsize=(9, 6)` or larger. Cramming a plot into the default 6.4x4.8 is what
  makes labels collide.
- `constrained_layout=True` (or call `fig.tight_layout()`) so nothing is clipped.
  Long axis labels and titles get cut off without it.
- Never go below **11pt** for tick labels or **12pt** for axis labels.
- Rotate long category labels instead of shrinking them:
  `ax.tick_params(axis="x", rotation=45)` with `ha="right"` alignment.
- If a legend crowds the data, move it out:
  `ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11)`.
- Many categories (feature importance, model comparison) read better as a
  horizontal bar chart — labels sit on the y-axis with room to breathe.
- One idea per axes. If you want four views, use
  `fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)`.

## Showing the plot

**Default to `plt.show()` — don't write a file unless the user asks.** The REPL
uses an interactive backend, so `plt.show()` opens the plot in a window. A
`savefig` leaves a stray PNG in the user's working directory; only save when they
ask to keep or share one, and then at a readable resolution:

```python
fig.savefig("pxr_parity.png", dpi=150, bbox_inches="tight")
```

`bbox_inches="tight"` is the savefig equivalent of the layout rules above —
without it, labels get cropped out of the saved image.

## Getting the data

Plot what the model actually produced rather than recomputing it:

```python
model = Model("pxr-reg-chemprop")
df = model.get_inference_predictions()      # has the target and prediction columns
fs = FeatureSet("aqsol_features")
df = fs.pull_dataframe()
```

Check the column names before plotting — don't guess which column holds the
prediction.

## Parallel coordinates for HPO trials

The default view of a hyperparameter search: one vertical axis per knob, one line per
trial, colored by the objective.

```python
from workbench.utils.plots import hpo

fig = hpo.parallel_coordinates(model)      # None if the model wasn't searched
fig.show()                                 # or fig.savefig(...) if asked
```

Only *adjacent* axes show their relationship, so axis order decides what the chart can
reveal — the function orders by `hpo_importance()`. See the `hpo` guide for reading the
numbers.

## Compound neighborhood graph

For a proximity result, `neighborhood.graph` renders the query at the center with
its closest neighbors around a ring — ring color = target value, edge width =
similarity — so an activity cliff jumps out.

```python
from workbench.utils.plots import neighborhood

fig = neighborhood.graph(query_id, nbrs, target_col="pec50")   # nbrs from prox.neighbors(...)
fig.show()                                                     # or fig.savefig(...) if asked
```

`nbrs` needs a `smiles` column and the query's own row (default `include_self=True`
provides it, similarity 1.0) — join SMILES from the FeatureSet if the proximity
result lacks it. See the `proximity` guide.

For an arbitrary set of structures with no query/neighbor relation (e.g. a
top-residuals panel), `vis.molecule_grid(smiles, captions, colors)` lays them out in
a captioned grid.

## PK profiles / concentration-time curves

A request for a **PK profile**, a **plasma concentration-time curve**, an **exposure
profile**, **Cp vs t**, or a **Bateman curve** all mean this plot. `pk.bateman` draws
the single oral dose, one-compartment model with first-order absorption and first-order
elimination, from a compound's volume of distribution and clearance.

Absorption rate `ka` is rarely pinned down by an early ADMET package, so it is a
**slider** rather than an argument — this is the one bundled plot that is Plotly,
because the slider needs it.

```python
from workbench.utils.plots import pk

fig = pk.bateman(volume=50, clearance=10, f_percent=40, dose=100)
fig.show()      # opens a browser tab; the slider needs that or a notebook
```

Volume and clearance usually come from an endpoint rather than a literal. A
multi-target endpoint names each head **`<target>_pred`** — `vd_pred`, `cl_pred` — while
a single-target one returns a bare `prediction`. Read the columns off the frame you got
back instead of assuming either shape:

```python
predictions = Endpoint("admet-meta").inference(df)
[c for c in predictions.columns if c.endswith("_pred")]     # what this endpoint actually returned

fig = pk.bateman(volume=predictions["vd_pred"].iloc[0], clearance=predictions["cl_pred"].iloc[0])
fig.show()
```

Watch the units when the inputs are predictions: ADMET models commonly report Vd in
**L/kg** and clearance in **mL/min/kg**, which do not agree. Convert clearance with
`* 0.06` to reach L/h/kg before passing both, or `ke` lands ~17x too fast.

A curve that comes out visibly flat or vertical means the V/CL pair is implausible, not
that the plot is wrong — the `pk_data` guide has the consistency checks.

Units are yours to choose as long as they agree — volume in L with clearance in L/h
gives `ke` in 1/h, and a dose in mg reads as mg/L. `ke = clearance / volume` and AUC
are derived and annotated; **AUC does not move with `ka`**, only the curve's shape
does, which is the point of dragging the slider.

## Highlighting and structure diffs

`img_from_smiles`, `svg_from_smiles`, and `molecule_grid` all take atom and bond
indices to highlight.

To answer "these two structures share a fingerprint but not a target value — what
actually differs?", `diff_molecules` draws them side by side with everything outside
their maximum common substructure highlighted:

```python
from workbench.utils.chem_utils.vis import diff_molecules, stereo_differences, structural_differences

fig = diff_molecules(smiles_a, smiles_b, captions=[id_a, id_b])
fig.show()
structural_differences(smiles_a, smiles_b)    # indices only: connectivity
stereo_differences(smiles_a, smiles_b)        # indices only: R/S centers and E/Z bonds
```

`diff_molecules` highlights both kinds, so **stereoisomers are marked at the
stereocenter or double bond** rather than coming back blank. Use the two index
functions when you want to say *which* kind it is: extra counterions and fragments
show up in `structural_differences`, enantiomers and geometry in
`stereo_differences`. See the `data_cleanup` guide.
