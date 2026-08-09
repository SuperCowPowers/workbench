# Plugin Pages

A complete, real-world Workbench plugin page to copy into your own plugin repo. The
`examples/plugins/` directory in the Workbench repo shows each plugin type in isolation;
this is one page built the way a production page is built — a page, its view, a custom
component, and clientside JS/CSS working together.

## The page

**Model Comparison** (`/model_comparison`) — a master-detail view of every published model
contest. A left rail lists contests grouped by pipeline; picking one shows the full
champion-vs-challenger report table with Δ-vs-champion coloring, and prediction plots for
the champion and the selected challenger.

```
plugin_pages/
  pages/model_comparison_page.py    # the page: layout + callbacks
  views/model_comparison_view.py    # data: read every /contests/* report
  components/prediction_plot.py     # a ModelPlot subclass with the chrome removed
  assets/mc/render.js               # clientside rail + comparison table
  assets/mc/styles.css              # page styling (scoped under .mc-root)
```

The contest reports come from the promotion arbiter, so the page needs published
`/contests/*` reports to show anything.

## Patterns worth stealing

**Clientside rendering off a single Store.** The view ships every contest report to the
browser once; `render.js` owns the rail, header, and table and redraws them on selection
with no server round-trip. Selection travels back to Dash via `set_props` on two Stores,
which drive the server-rendered plots. Only the plots — the genuinely expensive part — cost
a callback. Selecting a contest is 2 requests: one plot per side.

**Subclassing a built-in component.** `PredictionPlot` extends `ModelPlot` and swaps its two
sub-plugins for quieter ones. It inherits the model-type switch, multi-task target handling,
and theme re-rendering rather than reimplementing them.

**Memoized model construction.** `CachedModel` caches method results, not construction, so
its `__init__` is a full AWS metadata load. The page memoizes it keyed on the report
timestamp — cheap clicks, and a retrain still invalidates.

**A palette derived from Bootstrap.** `styles.css` defines its tokens as `var(--bs-body-color)`,
`var(--bs-tertiary-bg)` and friends, which the dashboard serves per request, so the page
follows any theme — the three built-ins and anything a customer adds — with no light/dark
branch of its own. Don't key off the `data-bs-theme` attribute for this; Bootstrap scopes its
palette to that attribute, so reading the variables is both simpler and correct.

**Re-rendering plots on a theme change.** CSS follows the theme by itself, but a Plotly figure
is baked at render time. `register_theme_callback()` from
`workbench.web_interface.utils.page_callbacks` re-renders every plugin on the page when the
theme store changes. A page that skips this keeps its plots on the old theme until a reload.

**Two-way deep links.** `?name=<endpoint>` selects a contest on load, and picking one in the
rail rewrites the query string, so the address bar is always copy-pasteable. Both directions
live in `render.js`: it reads the query string when nothing is selected yet, and writes with
`history.replaceState`. Don't write to `dcc.Location` — with `refresh="callback-nav"` that is
a navigation and remounts the page on every click. Artifact pages get the same behavior from
`sync_selection()` in `page_callbacks`.

## Using it

Copy the directories into your plugin repo, point `WORKBENCH_PLUGINS` at it, and run the
dashboard. Plugin type is determined by the directory, so keep the `pages/`, `views/`,
`components/`, and `assets/` layout.

The view and the component each run standalone for a quick check — the view prints a contest
summary, the component serves itself on http://localhost:8050:

```bash
python views/model_comparison_view.py
python components/prediction_plot.py
```

See https://supercowpowers.github.io/workbench/plugins/ for how plugins load, how `assets/`
is staged and served, and how to deploy to S3.
