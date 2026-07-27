# Plotting

> readable matplotlib plots, including labeled molecule structure grids

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

The REPL uses an interactive backend, so `plt.show()` opens a window. If the
user wants a file to keep or share, save it at a readable resolution:

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

**Build it from `model.hpo_search_space()`.** That is the plot's scaffold — one row per
knob is one axis, and each row already declares what the axis is: `dist` (`int` / `float`
/ `choice`), and in `spec` the bounds or the `options` and the `log` flag. Then
`model.hpo_results()["trials"]` supplies the lines to draw on it. See the `hpo` guide.

Taking the axes from the space rather than inferring them from the trial values is what
keeps a mixed-type knob from breaking the plot — and it keeps a knob on the chart even
when every trial happened to pick the same value.

What makes it readable:

- Axis per knob, typed from `dist`: a `choice` knob is categorical (rank its `options`),
  the rest are numeric. Log-scale where `spec` says `log`, and say so in the label.
- Scale each axis to the declared `low`/`high`, not the observed min and max. That is
  what makes a knob pinned against its own bound visible.
- Expand the `trials` frame's `hyperparameters` cell into one column per knob
  (`json.loads`) and join to the axes. The space describes the framework's full set and a
  search may have used a subset, so drive off the knobs the trials actually carry.
- Color by the objective with a **divergent** colormap centered on the baseline
  (`TwoSlopeNorm(vcenter=<baseline objective>)`), so hue answers the question that
  matters — did this trial beat the user's own hyperparameters — rather than where
  it ranks within the run. Set the direction so better-than-baseline gets the
  favorable hue; for a minimized objective that means reversing the map (`RdBu_r`).
- Draw the good trials thicker and more opaque on top of the faint ones, or the
  winners are lost in the crowd.
- Draw the two reference lines: the published config, and the `kind="baseline"`
  row (the user's own hyperparameters, scored on the same basis as the trials).
  A search plot without the baseline can't show whether the search achieved
  anything.

## Molecule structure panels

When the point is chemistry — an activity cliff, nearest neighbors, the top
residuals — render the actual structures in a labeled grid. Seeing near-identical
scaffolds with opposite activity tells the story a scatter plot can't.

`molecule_grid()` handles the layout (grid sizing, axis-off, invalid-SMILES gaps,
blank cells). You supply three parallel lists — SMILES, captions, and caption
colors — and it returns a matplotlib figure.

```python
from workbench.utils.chem_utils.vis import molecule_grid

smiles_col = next(c for c in df.columns if c.lower() == "smiles")   # see `compounds`
smiles = df[smiles_col].tolist()
captions = [f"{r.id}\npec50 = {r.pec50:.2f}" for r in df.itertuples()]
# Color captions by role so the pattern jumps out at a glance
colors = ["gold" if r.is_query else "#87d75f" if r.pec50 >= 5 else "salmon" for r in df.itertuples()]

fig = molecule_grid(smiles, captions, colors, suptitle="Activity cliff: OADMET-0002753 vs neighbors")
fig.show()                                          # or fig.savefig(path, dpi=150, bbox_inches="tight")
```

- **Caption every tile** with the id and the values that matter (activity,
  Tanimoto, predicted vs measured) — the structure alone isn't identifiable.
- **Color-code by role** (query vs active vs inactive, hit vs miss). Captions sit
  on the white figure, so any readable color works.
- The caller owns the domain choices (caption text, colors); `molecule_grid`
  just lays them out. It already guards unparseable SMILES.
