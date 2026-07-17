"""Plugin Page (Assets): demonstrates a plugin shipping clientside JS/CSS assets.

The plugin repo ships an assets/ folder (assets/hello/render.js + styles.css). The workbench
dashboard stages plugin assets/ into its own assets tree, so Dash serves them (at
/assets/plugins/hello/...) and injects the <script>/<link> into every page head. That makes
the "hello" namespace available to a clientside callback, exactly like the app's own assets.

Pattern: a server-side callback fills a dcc.Store; a clientside callback (namespace "hello")
owns the rendering. Python fills data, JS owns pixels.
"""

import dash
from dash import Input, Output, callback, clientside_callback, ClientsideFunction, html, dcc, register_page

# Workbench Imports
from workbench.cached.cached_meta import CachedMeta


class PluginPageAssets:
    """Plugin Page that renders via a clientside function shipped in the plugin's assets/."""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.page_name = "Plugin Assets Demo"
        self.meta = CachedMeta()

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""
        register_page(
            __file__,
            path="/plugin_assets",
            name=self.page_name,
            layout=self.page_layout(),
        )
        self.page_callbacks()

    def page_layout(self) -> dash.html.Div:
        """Set up the layout for the page"""
        return html.Div(
            children=[
                html.H1(self.page_name),
                # Server fills this Store; the clientside "hello" renderer reads it
                dcc.Store(id="hello-data"),
                # The clientside function renders into this root div
                html.Div(id="hello-root"),
                # Hidden output for the render callback's status string
                html.Div(id="hello-render-signal", style={"display": "none"}),
                # Interval that triggers once on page load
                dcc.Interval(id="hello-page-load", interval=100, max_intervals=1),
            ]
        )

    def page_callbacks(self):
        """Set up the callbacks for the page"""

        # Server-side: fill the Store with data (Python owns the data)
        @callback(
            Output("hello-data", "data"),
            Input("hello-page-load", "n_intervals"),
        )
        def _fill_store(_n):
            models = self.meta.models()
            names = models["Model Group"].tolist()[:8] if models is not None and not models.empty else []
            return {"greeting": f"Hello from a plugin asset! ({len(names)} models)", "items": names}

        # Clientside: render the Store data (JS from assets/hello/render.js owns the pixels)
        clientside_callback(
            ClientsideFunction(namespace="hello", function_name="render"),
            Output("hello-render-signal", "children"),
            Input("hello-data", "data"),
        )
