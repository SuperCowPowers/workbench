"""A Plugin Page for Model Comparison.

Master-detail: a compact contest picker (left rail) and the selected contest's full
champion-vs-challenger report table, with a prediction plot (confusion matrix for
classifiers, scatter for regressors) for the champion (top row) and the selected
challenger (bottom row).

The rail and comparison table are rendered clientside (assets/mc/render.js + styles.css)
off a single Store holding every /contests/* report -- no server round-trip on selection.
The two plots are server-rendered workbench components, driven by the clientside
selection via two Stores that render.js writes with set_props.
"""

from functools import lru_cache

import dash
from dash import dcc, html, no_update, register_page
from dash.dependencies import Input, Output, State, ClientsideFunction

# Workbench Imports
from workbench.cached.cached_model import CachedModel
from workbench.utils.plugin_manager import PluginManager
from workbench.web_interface.utils.page_callbacks import register_theme_callback

INFERENCE_RUN = "full_cross_fold"


@lru_cache(maxsize=16)
def _model(name: str, _timestamp: str) -> CachedModel:
    """Build a CachedModel, memoized.

    CachedModel caches *method results*, not construction -- its __init__ is a full
    uncached AWS metadata load. Keyed on the report timestamp, so a retrain (which
    republishes the report) builds a fresh model rather than serving stale metadata.

    Args:
        name (str): The model name
        _timestamp (str): The report's publish time, for cache invalidation only

    Returns:
        CachedModel: The model
    """
    return CachedModel(name)


def _contest(data: list, endpoint: str) -> dict | None:
    """The contest entry for an endpoint, from the Store data."""
    if not data or not endpoint:
        return None
    return next((c for c in data if c.get("endpoint") == endpoint), None)


def _timestamp(contest: dict | None) -> str:
    """The contest's publish timestamp (for the _model cache key)."""
    rows = contest.get("rows") if contest else None
    return str(rows[0].get("timestamp")) if rows else ""


class ModelComparisonPage:
    """A Plugin Page for Model Comparison"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.app = None
        self.plots = {}  # "champion"/"challenger" -> PredictionPlot

        # Get our view and custom component from the PluginManager
        pm = PluginManager()
        self.view = pm.get_view("ModelComparisonView")
        self.prediction_plot_class = pm.get_all_plugins()["components"]["PredictionPlot"]

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""
        self.app = app

        # A prediction plot (confusion matrix / scatter) per side
        for side in ["champion", "challenger"]:
            self.plots[side] = self.prediction_plot_class()

        register_page(__name__, path="/model_comparison", name="Model Comparison", layout=self.page_layout())

        self.data_callback()
        self.render_callback()
        self.champion_callback()
        self.challenger_callback()

        # The page's CSS follows the theme on its own; the plots need a re-render
        register_theme_callback(list(self.plots.values()))

    def page_layout(self) -> html.Div:
        """Set up the layout for the page"""
        return html.Div(
            # Breathing room from the window edges (tighter on the left so the rail sits closer)
            style={"margin": "30px 30px 30px 16px"},
            children=[
                # One-shot load trigger; the Stores below feed the clientside renderer.
                dcc.Input(id="mc_load", type="hidden", value="load"),
                dcc.Store(id="mc_store"),  # every /contests/* report
                dcc.Store(id="mc_selected_endpoint"),  # picked contest (set clientside)
                dcc.Store(id="mc_selected_challenger"),  # picked challenger model (set clientside)
                html.Div(
                    className="mc-root",
                    children=[
                        html.Div(
                            className="mc-header",
                            children=[
                                html.H3("Model Comparison"),
                                # Framework legend, populated clientside from the frameworks present
                                html.Div(id="mc-legend", className="mc-legend"),
                            ],
                        ),
                        html.Div(
                            className="mc-grid",
                            children=[
                                html.Div(id="mc-rail", className="mc-rail"),
                                html.Div(
                                    className="mc-main",
                                    children=[
                                        html.Div(id="mc-head", className="mc-head"),
                                        html.Div(id="mc-table"),
                                        html.Div(
                                            className="mc-plot-region",
                                            children=[self._plot_row("champion"), self._plot_row("challenger")],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                # Dummy output target for the clientside render callback
                html.Div(id="mc-render-signal", style={"display": "none"}),
            ],
        )

    def _plot_row(self, side: str) -> html.Div:
        """One plot row: a labeled header + the prediction plot.

        Args:
            side (str): "champion" or "challenger"

        Returns:
            html.Div: The plot row
        """
        return html.Div(
            className="mc-prow",
            children=[
                html.Div(
                    className="mc-prow-head",
                    children=[
                        html.Span(side, className=f"mc-tag mc-tag-{side}"),
                        html.Span(id=f"mc-{side}-name", className="mc-prow-name"),
                        # The model name is written clientside; the plot callback fills this
                        # span with a status message when there's nothing to plot.
                        html.Span(id=f"mc-{side}-status", className="mc-prow-status"),
                    ],
                ),
                # The plots take seconds to load (model construction + inference pull); the
                # spinner keeps the previous model's figures from showing under the new row.
                dcc.Loading(html.Div(self.plots[side].create_component(f"mc_model_plot_{side}"), className="mc-plot")),
            ],
        )

    def data_callback(self):
        """Load every contest report into the Store, once on page open"""

        @self.app.callback(Output("mc_store", "data"), Input("mc_load", "value"))
        def load(_trigger):
            self.view.refresh()
            return self.view.view_data()

    def render_callback(self):
        """Clientside: draw the rail + table and label the plot rows from the Store + selection"""
        self.app.clientside_callback(
            ClientsideFunction(namespace="mc", function_name="render"),
            Output("mc-render-signal", "children"),
            Input("mc_store", "data"),
            Input("mc_selected_endpoint", "data"),
            Input("mc_selected_challenger", "data"),
        )

    def _outputs(self, side: str) -> list:
        """Callback outputs for a side, in the order _side returns them.

        Args:
            side (str): "champion" or "challenger"

        Returns:
            list: [*prediction Outputs, status Output]
        """
        outputs = [Output(c, p) for c, p in self.plots[side].properties]
        return outputs + [Output(f"mc-{side}-status", "children")]

    def champion_callback(self):
        """Update the champion's prediction plot when the contest changes"""

        @self.app.callback(self._outputs("champion"), Input("mc_selected_endpoint", "data"), State("mc_store", "data"))
        def update_champion(endpoint, data):
            contest = _contest(data, endpoint)
            name = next((r["model"] for r in contest["rows"] if r.get("role") == "champion"), None) if contest else None
            if not name:
                return self._side("champion", None, "Select a contest.")
            return self._side("champion", _model(name, _timestamp(contest)))

    def challenger_callback(self):
        """Update the challenger's prediction plot when the selected challenger changes"""

        @self.app.callback(
            self._outputs("challenger"),
            Input("mc_selected_challenger", "data"),
            State("mc_selected_endpoint", "data"),
            State("mc_store", "data"),
        )
        def update_challenger(name, endpoint, data):
            if not name:
                return self._side("challenger", None, "No challenger for this contest.")
            return self._side("challenger", _model(name, _timestamp(_contest(data, endpoint))))

    def _side(self, side: str, model: CachedModel | None, message: str = "") -> list:
        """Outputs for one side: the prediction plot properties plus the header status text.

        The prediction plot morphs by model type, so plotting needs no type check. With no
        model it's a composite of visibility styles and two sub-plots, so leave its
        properties untouched rather than synthesizing a value per slot, and put the message
        in the row header instead.

        Args:
            side (str): "champion" or "challenger"
            model (CachedModel | None): The model to plot, or None for nothing to show
            message (str): The status message, when there's no model

        Returns:
            list: [*prediction properties, status text]
        """
        plot = self.plots[side]
        if model is None:
            return [no_update] * len(plot.properties) + [message]
        return plot.update_properties(model, inference_run=INFERENCE_RUN) + [""]
