# Building a Dashboard Plugin Page

> write a custom dashboard page: layout, view, callbacks, theming, deep links

A plugin page is a full route in the dashboard — your own layout, data, and
callbacks. Point `WORKBENCH_PLUGINS` at a directory and everything under it loads
at startup. See the `dashboard` guide for the dashboard itself; this is for
building a page.

## Start from the exemplar

A complete working page ships with the package. Read it before writing anything,
and copy it rather than starting from scratch:

```python
import workbench, pathlib
EX = pathlib.Path(workbench.__file__).parent / "plugin_pages"
print((EX / "README.md").read_text())
```

Pages, views, and components go in directories named for their type. A page
resolves its view and components through `PluginManager` by class name instead of
importing them, which is what lets a client swap in their own implementation.

## Traps

None of these are guessable, and each one costs an hour.

**Theme the CSS off Bootstrap's variables** — `var(--bs-body-color)`,
`var(--bs-tertiary-bg)`, `var(--bs-border-color)`. The dashboard serves the right
Bootstrap build per request, so the page follows every theme, customer ones
included, with no light/dark branch of its own. Do not write
`:root[data-bs-theme="light"]` overrides against a parallel palette.

**Plotly figures do not re-theme themselves.** CSS follows the theme; a figure is
baked when it is built. `register_theme_callback([plot_a, plot_b])` from
`workbench.web_interface.utils.page_callbacks` re-renders plugins, which works
because a plugin caches its input. A plain `ComponentInterface` component caches
nothing, so instead give the callback that already builds its figure an extra
`Input` on `page_callbacks.THEME_STORE_ID`.

**Never write to `dcc.Location`.** It runs with `refresh="callback-nav"`, so a
write is a navigation that remounts the whole page. Rewrite the query string with
`history.replaceState` instead. For a page built on an `AGTable`,
`sync_selection("my_table", "my_page_loaded")` wires `?name=` in both directions.

**`CachedModel` caches method results, not construction.** Its `__init__` is a full
AWS metadata load, so building one per callback is slow. Memoize it.

**Plugin assets are staged at startup**, so editing a `.js` or `.css` file needs a
dashboard restart, not a browser reload.
