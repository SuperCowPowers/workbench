# Interactive Apps

> Dash apps served from the REPL — the PK explorer, and how to adapt one

`workbench.utils.plots` returns figures. `workbench.utils.apps` serves **apps**: a Dash
UI on a background thread, returning the URL. Reach for one only when hover has to show
something a tooltip cannot — a molecule, a table, a second plot. A `hovertemplate` will
not render an `<img>`, which is the whole reason these exist.

## The PK explorer

One curve per compound, colored by group, structure panel on hover.

```python
from workbench.utils import apps

apps.pk_explorer(df, cl_col="cl", vd_col="vd", smiles_col="smiles", id_col="id")
```

Rows missing clearance or volume of distribution are dropped and the count is logged —
report it to the user, since a third of a set can vanish silently otherwise.

Pass `half_life_col` when the frame carries a reported half-life. The derived and
reported values are correlated and logged; below 0.95 means the units disagree, and the
usual cause is clearance in mL/min/kg being treated as L/h/kg.

## What is assumed, and what that costs

The curves are **oral** Bateman profiles. An IV frame carries neither an absorption rate
nor a bioavailability, so `ka = 1.0 /h` and `f_percent = 100` are invented defaults and
both are printed in the plot title. Say so when presenting one.

Moving `ka` slides every tmax and changes no AUC. A long tmax at the slow end is
flip-flop kinetics — absorption rate-limiting because `ke << ka` — not a claim about
absorption in the data.

Two judgment calls the app makes that you may want to revisit with the user:

- **The y-axis floors three decades below the tallest peak.** Bateman curves decay
  forever, so the bottom of a log axis is arithmetic rather than measurement —
  concentrations below the assay's limit of quantitation are fiction. Pass `y_floor`
  when a real LLOQ is known.
- **Do not normalize curves by Cmax.** The family collapses into a function of half-life
  alone and the plot stops saying anything.

## Grouping

`group_col` colors by any column — chemotype, project, potency bin. Omit it and the app
clusters on the standardized (log CL, log Vd) plane, `n_clusters=4`, relabeled so group 0
has the longest median half-life.

Cluster on the **parameter plane, not the curves**. Two groups can share a half-life and
still separate cleanly on volume of distribution; profile space cannot see that.

## Adapting one

`pk_explorer` is `build_app` plus `serve`. For anything custom, build the app, change it,
then serve it yourself:

```python
from workbench.utils.apps import build_app
from workbench.utils.apps._serve import serve

app = build_app(df, group_col="series", duration=48)
serve(app)
```

For a different app entirely, read the source and adapt it — `inspect.getsource` on
`build_app` (see the `introspection` guide). Two things there are not obvious and cost a
round trip each if rediscovered:

- **A callback must never write the container holding its other outputs.** Writing the
  wrapping `Div` replaces those components with id-less copies, so every hover after the
  first fires at outputs that no longer exist — the panel updates exactly once, then
  silently stops. Give each leaf its own id and write only the leaves.
- **A live Dash app rejects redefined callbacks.** Fixing one means building a fresh app,
  which is why `serve` takes a new port each time and leaves the old server on its thread.

Structure images come from `svg_from_smiles`, which returns a ready
`data:image/svg+xml;base64,...` URI for `html.Img(src=...)`.
