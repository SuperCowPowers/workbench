# Plotting

> readable matplotlib plots, including molecule structure and neighborhood views

Use **matplotlib**. Make sure text, legends, and axis labels have enough space to
be readable — a cramped plot is a useless plot.

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
df = fs.pull_dataframe()                # or fs.query(...) if it is large
```

Check the column names before plotting — don't guess which column holds the
prediction.

## Common plots

- **Parity (predicted vs actual):** scatter plus a `y = x` reference line. Set
  `ax.set_aspect("equal")` and match the axis limits, otherwise the diagonal
  lies about the fit.
- **Residuals:** residual against predicted, with `ax.axhline(0)`. Reveals bias
  and heteroskedasticity that a parity plot hides.
- **Distributions:** `ax.hist(..., bins=50)`, or overlay train vs holdout to
  check for drift.
- **Model comparison:** horizontal bar chart of a metric across models, sorted.

For large scatter plots use `alpha=0.3` and `s=10` so the dense regions stay
readable instead of turning into a solid block.

## Parallel coordinates for HPO trials

The default view of a hyperparameter search: one vertical axis per parameter, one
line per trial, colored by the objective. It shows which regions the good trials
cluster in and lets you trace a single config across every axis.

Three methods build it, and each answers a different question (see the `hpo` guide):

```python
model.hpo_search_space()   # the axes: what each knob is and what its range means
model.hpo_results()        # the lines: one per trial, plus the baseline
model.hpo_importance()     # the axis ORDER, and which axes are worth keeping
```

Taking the axes from the space rather than inferring them from the trial values is what
keeps a mixed-type knob from breaking the plot — and it keeps a knob on the chart even
when every trial happened to pick the same value.

What makes it readable:

- **Order the axes by `importance`, most important on the left.** Only *adjacent* axes show
  their relationship in a parallel-coordinates plot, so axis order decides what the chart
  can reveal; an arbitrary order buries the structure.
- Axis per knob, typed from `dist`: a `choice` knob is categorical (rank its `options`),
  the rest are numeric. Log-scale where `spec` says `log`, and say so in the label.
- Scale each axis to the declared `low`/`high`, not the observed min and max. That is
  what makes a knob pinned against its own bound visible.
- Clip every line to the axis range (`[0, 1]` normalized). A trial or reference value
  outside a knob's declared bounds would otherwise draw off the axes.
- Expand the `trials` frame's `hyperparameters` cell into one column per knob
  (`json.loads`) and join to the axes. The space describes the framework's full set while a
  search may have used a subset — the `importance` frame is exactly that subset, so driving
  the axis list off it gets the right knobs and the right order in one step.
- Color by the objective with a **divergent** colormap centered on the baseline
  (`TwoSlopeNorm(vcenter=<baseline objective>)`), so hue answers the question that
  matters — did this trial beat the user's own hyperparameters — rather than where
  it ranks within the run. Set the direction so better-than-baseline gets the
  favorable hue; for a minimized objective that means reversing the map (`RdBu_r`).
- **Every trial line at `alpha=0.3`.** Only two lines break that: the `kind="baseline"` row
  (the user's own hyperparameters, scored on the same basis as the trials) and the winner.
  Draw those last, opaque and thicker, so they read on top of the crowd. A search plot
  without the baseline can't show whether the search achieved anything.

## Compound neighborhood graph

For a proximity result, `neighborhood_graph` renders the query at the center with
its closest neighbors around a ring — ring color = target value, edge width =
similarity — so an activity cliff jumps out.

```python
from workbench.utils.chem_utils.vis import neighborhood_graph

fig = neighborhood_graph(query_id, nbrs, target_col="pec50")   # nbrs from prox.neighbors(...)
fig.show()                                                     # or fig.savefig(...) if asked
```

`nbrs` needs a `smiles` column and the query's own row (default `include_self=True`
provides it, similarity 1.0) — join SMILES from the FeatureSet if the proximity
result lacks it. See the `proximity` guide.

For an arbitrary set of structures with no query/neighbor relation (e.g. a
top-residuals panel), `vis.molecule_grid(smiles, captions, colors)` lays them out in
a captioned grid.
