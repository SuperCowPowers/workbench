# Building a Dashboard Plugin Page

> write a custom dashboard page: layout, view, callbacks, theming, deep links

A plugin page is a full route in the Workbench Dashboard — your own layout, your
own data, your own callbacks. Point `WORKBENCH_PLUGINS` at a directory and the
dashboard loads everything under it at startup. This is how clients add
business-specific views without forking the dashboard.

For what the dashboard *is* and how to link to its existing pages, see the
`dashboard` guide. This one is for building a new page.

## Read the exemplar first

A complete, working page ships with the package. Read it before writing anything
— it is the current, correct shape, and it beats reciting patterns from memory:

```python
import workbench, pathlib
EX = pathlib.Path(workbench.__file__).parent / "plugin_pages"

print((EX / "README.md").read_text())                       # what it does and why
print((EX / "pages" / "model_comparison_page.py").read_text())
```

```
plugin_pages/
  README.md                         # the patterns, and the reasoning behind them
  pages/model_comparison_page.py    # the page: layout + callbacks
  views/model_comparison_view.py    # data: read the reports behind the page
  components/prediction_plot.py     # subclassing a built-in component
  assets/mc/render.js               # clientside rendering + deep links
  assets/mc/styles.css              # theming that follows the dashboard
```

Copy that directory into the user's plugin repo as a starting point rather than
writing a page from scratch.

## Layout of a plugin directory

Plugin type is determined by the **directory**, so the names matter:

```
plugins/
  pages/        # a class with page_setup(app) -- registers its own route
  views/        # subclass PageView -- reshape the data behind a page
  components/   # subclass PluginInterface -- also auto-loads onto artifact pages
  assets/       # clientside JS/CSS, staged and injected by Dash
  packages/     # importable Python packages for your plugins
```

## The page class

A page is a plain class with `page_setup(app)`. The dashboard instantiates it and
calls that method; you register the route and the callbacks inside:

```python
class MyPage:
    def __init__(self):
        pm = PluginManager()
        self.view = pm.get_view("MyView")
        self.plot = pm.get_all_plugins()["components"]["MyPlot"]()

    def page_setup(self, app):
        self.app = app
        register_page(__name__, path="/my_page", name="My Page", layout=self.page_layout())
        self.data_callback()
        register_theme_callback([self.plot])
```

Views and components are looked up through `PluginManager` by class name, not
imported directly — that is what lets a client swap an implementation.

## Things that are easy to get wrong

**Theming: derive the palette from Bootstrap's variables.** Define your CSS tokens
as `var(--bs-body-color)`, `var(--bs-tertiary-bg)`, `var(--bs-border-color)`. The
dashboard serves the right Bootstrap build per request, so the page follows any
theme, including customer-defined ones, with no light/dark branch of your own. Do
**not** write `:root[data-bs-theme="light"]` overrides and maintain a parallel
palette — Bootstrap already scopes its palette to that attribute.

**Plotly figures need an explicit re-render on theme change.** CSS follows the
theme by itself; a figure is baked when it is built. Register this or your plots
stay on the old theme until a full reload:

```python
from workbench.web_interface.utils.page_callbacks import register_theme_callback

register_theme_callback([plot_a, plot_b])   # PluginInterface instances
```

That works for plugins because they cache their input and can rebuild. A plain
`ComponentInterface` component holds nothing between callbacks, so re-render it by
adding the theme store as an extra `Input` on the callback that already builds its
figure — the store id is `page_callbacks.THEME_STORE_ID`.

**Deep links: rewrite the query string, never write to `dcc.Location`.** The
dashboard's Location runs with `refresh="callback-nav"`, so writing to it is a
navigation that remounts the page on every selection. Use
`history.replaceState` from clientside JS. For a page built on an `AGTable`, the
whole two-way behavior is one call:

```python
from workbench.web_interface.utils.page_callbacks import sync_selection

sync_selection("my_table", "my_page_loaded")   # ?name=<artifact> <-> selected row
```

**`CachedModel` caches method results, not construction.** Its `__init__` is a full
uncached AWS metadata load, so building one inside a callback that fires on every
click is slow. Memoize with `functools.lru_cache`, keyed on something that changes
when the model does.

**Keep selection off the server.** If a page redraws a table or list on every click,
render it clientside from a single `dcc.Store` and push the selection back with
`set_props`. Reserve server callbacks for the genuinely expensive parts, like
building a figure from inference data.

## Running it

```bash
WORKBENCH_PLUGINS=/path/to/plugins python applications/aws_dashboard/app.py
```

Views and components usually have a `__main__` block so they run standalone — a
view prints its data, a component serves itself on http://localhost:8050 through
`PluginUnitTest`. That is much faster than restarting the whole dashboard.

Plugin assets are staged into the dashboard's asset tree **at startup**, so edits
to `.js` or `.css` need a dashboard restart, not just a browser reload.

## More

- Plugins overview: https://supercowpowers.github.io/workbench/plugins/
- S3 plugins: https://supercowpowers.github.io/workbench/admin/dashboard_s3_plugins/
