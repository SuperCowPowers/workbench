!!! tip inline end "Workbench Plugins"
    The Workbench toolkit provides a flexible plugin architecture to expand, enhance, or even replace the [Dashboard](../aws_setup/dashboard_stack.md). Make custom UI components, views, and entire pages with the plugin classes described here.

The Workbench Plugin system lets you customize how your AWS Machine Learning Pipeline is displayed, analyzed, and visualized. An easy-to-use Python API enables developers to make new [Dash/Plotly](https://plotly.com/) components, data views, and entirely new web pages focused on business use cases.

### Concept Docs
- [Workbench Plugin Overview](https://docs.google.com/presentation/d/1RjpMmJW1i9auPztn2xXYmYKXsZjsnG7vVaCQQ4FLIMM/edit?usp=sharing)

## Plugin types

Point `WORKBENCH_PLUGINS` at a directory (local path or `s3://...`) and Workbench loads everything below at startup. See the [full example plugins](https://github.com/SuperCowPowers/workbench/tree/main/examples/plugins).

```
plugins/
  components/   # Web components: subclass PluginInterface, auto-load onto an artifact page
  pages/        # Full pages: a class with page_setup(app); registers its own route
  views/        # Page views: subclass PageView, reshape the data behind a page
  assets/       # Clientside JS/CSS — served + injected by Dash (see below)
  packages/     # Importable Python packages (added to PYTHONPATH for your plugins)
```

## Make a component

Each component plugin inherits from `PluginInterface`, sets two class attributes, and implements two methods. These are validated during tests and at runtime.

**Note:** For full code see the [Model Plugin Example](https://github.com/SuperCowPowers/workbench/blob/main/examples/plugins/components/model_plugin.py).

```python
class ModelPlugin(PluginInterface):
    """A Model Plugin Component"""

    # Where to auto-load, and what object update_properties receives
    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create the container for this component"""
        self.component_id = component_id
        self.container = dcc.Graph(id=component_id, ...)
        self.properties = [(self.component_id, "figure")]  # (id, prop) driven by callbacks
        return self.container

    def update_properties(self, model: Model, **kwargs) -> list:
        """Return updated values, one per entry in self.properties"""
        pie_figure = go.Figure(data=..., ...)
        return [pie_figure]
```

**Required attributes**

- `auto_load_page` — which artifact page auto-loads the component: `PluginPage.DATA_SOURCE`, `FEATURE_SET`, `MODEL`, `ENDPOINT`, `GRAPH`, or `COMPOUND`. Use `CUSTOM`/`NONE` to opt out of auto-loading (e.g. a component you place on your own plugin page).
- `plugin_input_type` — the object handed to `update_properties`: `PluginInputType.MODEL`, `ENDPOINT`, `DATAFRAME`, etc.

## Clientside assets (JS/CSS)

Drop `.js`/`.css` anywhere under the top-level `assets/` folder. Workbench stages them into the Dashboard's assets tree, so Dash serves them and injects `<script>`/`<link>` into every page head — the same treatment the app's own assets get.

```
assets/hello/render.js   ->  /assets/plugins/hello/render.js   (<script> injected)
assets/hello/styles.css  ->  /assets/plugins/hello/styles.css  (<link> injected)
```

Register a namespace in your JS and wire it from a page with `ClientsideFunction`:

```javascript
// assets/hello/render.js
window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.hello = { render: function (data) { /* ...owns the pixels... */ } };
```

```python
clientside_callback(
    ClientsideFunction(namespace="hello", function_name="render"),
    Output("hello-render-signal", "children"),
    Input("hello-data", "data"),
)
```

Namespace your CSS class names (injected CSS is global) and co-locate each page's JS/CSS in `assets/<namespace>/`. See [`plugin_page_assets.py`](https://github.com/SuperCowPowers/workbench/blob/main/examples/plugins/pages/plugin_page_assets.py) for a complete working example.

## Deployment

- **S3 (recommended for iteration):** set `WORKBENCH_PLUGINS` to `s3://my-bucket/workbench_plugins` and copy your plugins up. Full walkthrough: [Dashboard with S3 Plugins](../admin/dashboard_s3_plugins.md).
- **Local dev:** set `WORKBENCH_PLUGINS` to a local directory, or run a page module directly (each example has a `__main__` block).

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

Need help with plugins? Want a customized application tailored to your business needs?

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
