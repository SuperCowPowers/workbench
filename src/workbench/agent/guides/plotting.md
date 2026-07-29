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

The default view of a hyperparameter search: one vertical axis per knob, one line per
trial, colored by the objective. Shows which regions the good trials cluster in and lets
you trace a single config across every axis.

```python
from workbench.utils.hpo_plots import hpo_parallel_coordinates

fig = hpo_parallel_coordinates(model)                    # None if the model wasn't searched
fig.show()                                               # or fig.savefig(...) if asked
```

Two knobs worth knowing, for when the user asks:

- `use_curves=False` — straight segments instead of curves. Lines curve by default, flat
  where they cross each axis; some people prefer the plain polyline.
- `completed_only=True` — leave pruned trials off. They are drawn by default, since where
  the search looked is part of the picture.

It handles the parts that are easy to get subtly wrong: axes ordered by
`hpo_importance()` (only *adjacent* axes show their relationship, so the order decides
what the chart can reveal), each scaled to the knob's declared bounds and clipped, color
centered on the baseline so hue answers "did this trial beat the user's own
hyperparameters", and the baseline and published config drawn as reference lines. The
color scale is set by the completed trials, so a trial pruned early can't flatten it. See
the `hpo` guide for reading the numbers.

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

## Highlighting and structure diffs

`img_from_smiles` and `svg_from_smiles` take `highlight_atoms` / `highlight_bonds`
(atom and bond indices) plus a `highlight_color`. `svg_from_smiles(..., encode=False)`
returns raw SVG markup instead of a base64 data URI, for embedding in a report.

To answer "these two structures share a fingerprint but not a target value — what
actually differs?", `diff_molecules` draws them side by side with everything outside
their maximum common substructure highlighted:

```python
from workbench.utils.chem_utils.vis import diff_molecules, structural_differences

fig = diff_molecules(smiles_a, smiles_b, captions=[id_a, id_b])
fig.show()
atoms, bonds = structural_differences(smiles_a, smiles_b)         # indices only
```

`molecule_grid` takes the same `highlight_atoms` / `highlight_bonds` (one list per
molecule) when you want highlighting across a larger panel.

Read an **empty** highlight as a finding rather than a failure: MCS matches
connectivity, so a pair that differs only in stereochemistry or double-bond geometry
highlights nothing. That distinguishes the two big classes of coincident pair — extra
counterions and fragments light up, stereoisomers stay blank. See the `data_cleanup`
guide.
