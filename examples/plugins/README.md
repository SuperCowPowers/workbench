# Workbench Plugins

Customize the Workbench Dashboard with your own Dash/Plotly components, views, and pages.
Point `WORKBENCH_PLUGINS` at this directory (local path or `s3://...`) and Workbench loads
everything below at startup.

## Directory layout

```
plugins/
  components/   # Web components: subclass PluginInterface, auto-load onto an artifact page
  pages/        # Full pages: a class with page_setup(app); registers its own route
  views/        # Page views: subclass PageView, reshape the data behind a page
  assets/       # Clientside JS/CSS (see below) — served + injected by Dash
  packages/     # Importable Python packages (added to PYTHONPATH for your plugins)
  branding/     # Dashboard branding + Plotly templates
```

Only the subdirs you need are required; a plugin repo can be a single `components/` file.

## Clientside assets (JS/CSS)

Drop `.js`/`.css` anywhere under the top-level `assets/` folder. Workbench stages them into
the Dashboard's own assets tree, so Dash serves them and injects `<script>`/`<link>` into every
page head automatically — the same treatment the app's own assets get.

```
assets/hello/render.js     ->  served at /assets/plugins/hello/render.js  (<script> injected)
assets/hello/styles.css    ->  served at /assets/plugins/hello/styles.css (<link> injected)
```

Your JS registers a namespace, and a page wires it with `ClientsideFunction`:

```javascript
// assets/hello/render.js
window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.hello = { render: function (data) { /* ...owns the pixels... */ } };
```

```python
# pages/plugin_page_assets.py
clientside_callback(
    ClientsideFunction(namespace="hello", function_name="render"),
    Output("hello-render-signal", "children"),
    Input("hello-data", "data"),
)
```

Conventions: co-locate each page's JS/CSS in `assets/<namespace>/`, and namespace your CSS
class names (`.hello-card`) — injected CSS is global. See `pages/plugin_page_assets.py` +
`assets/hello/` for a complete working example.

## Local testing

Set `WORKBENCH_PLUGINS` to this directory and run the dashboard, or run a single page module
directly (each has a `__main__` block):

```bash
python pages/plugin_page_2.py
```

## Deploying to S3

See [Dashboard with S3 Plugins](../../docs/admin/dashboard_s3_plugins.md). Copy `*.py` **and**
your `assets/` files up to the bucket — a `--include "*.py"`-only copy silently drops JS/CSS.
