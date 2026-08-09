# Plugin Pages

A complete, real-world Workbench plugin page you can copy into your own plugin repo.
Where [`examples/plugins/`](../examples/plugins) shows each plugin type in isolation, this
is one page built the way a production page is built: a page, its view, a custom component,
and clientside JS/CSS working together.

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
a callback.

**Subclassing a built-in component.** `PredictionPlot` extends `ModelPlot` and swaps its two
sub-plugins for quieter ones. It inherits the model-type switch, multi-task target handling,
and theme re-rendering rather than reimplementing them.

**Memoized model construction.** `CachedModel` caches method results, not construction, so
its `__init__` is a full AWS metadata load. The page memoizes it keyed on the report
timestamp — cheap clicks, and a retrain still invalidates.

**Theme-aware CSS.** Every color is a custom property with light-mode overrides under
`:root[data-bs-theme="light"]`, so the page follows the dashboard's theme toggle.

**Two-way deep links.** `?name=<endpoint>` selects a contest on load, and picking one in the
rail rewrites the query string, so the address bar is always copy-pasteable. A server
callback handles the load direction; `render.js` handles the write with `history.replaceState`
— writing to `dcc.Location` would be a navigation and would remount the page on every click.
The artifact pages get the same behavior from `workbench.web_interface.utils.url_sync`.

## Using it

Copy the directories into your plugin repo, point `WORKBENCH_PLUGINS` at it, and run the
dashboard. Plugin type is determined by the directory, so keep the `pages/`, `views/`,
`components/`, and `assets/` layout.

The view and the component each run standalone for a quick check — the view prints a contest
summary, the component serves itself on http://localhost:8050:

```bash
python plugin_pages/views/model_comparison_view.py
```

See [`examples/plugins/README.md`](../examples/plugins/README.md) for how plugins load, how
`assets/` is staged and served, and how to deploy to S3.
