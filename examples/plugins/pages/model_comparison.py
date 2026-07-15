"""Model Comparison Page: champion/challenger contests with metrics and prediction overlays.

Layout:
    [ regression champions ]  [ classification champions ]
    [ ranked challengers   ]  [ comparison metrics       ]
                              [ comparison plot          ]

Selecting a champion (either table) populates the ranked challenger table; selecting a
challenger shows the champion/challenger/delta metrics and the prediction overlay plot.
"""

import sys
from pathlib import Path

import dash
from dash import html, dcc, page_container, register_page, callback, ctx, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Workbench Imports
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.web_interface.components.plugins.comparison_plot import ComparisonPlot

# Plugin View Import (same repo layout as the other page examples)
sys.path.append(str(Path(__file__).parent.parent / "views"))
from model_comparison_view import ModelComparisonView  # noqa: E402


class ModelComparisonPage:
    """Plugin Page: Champion/Challenger Model Comparison"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.page_name = "Model Comparison"
        self.view = ModelComparisonView()

        # Components (created in page_setup)
        self.reg_champions = AGTable()
        self.class_champions = AGTable()
        self.challengers = AGTable()
        self.metrics = AGTable()
        self.plot = ComparisonPlot()
        self.components = {}

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""

        # Create the components
        self.components = {
            "reg_champions": self.reg_champions.create_component("mc_reg_champions", max_height=300),
            "class_champions": self.class_champions.create_component("mc_class_champions", max_height=300),
            "challengers": self.challengers.create_component("mc_challengers", max_height=400),
            "metrics": self.metrics.create_component("mc_metrics", max_height=150),
            "plot": self.plot.create_component("mc_comparison_plot"),
        }

        # Register this page with Dash and set up the layout
        register_page(
            __file__,
            path="/model_comparison",
            name=self.page_name,
            layout=self.page_layout(),
        )

        # Callbacks
        self.page_load_callbacks()
        self.page_callbacks()
        self.plot.register_internal_callbacks()

    def page_layout(self) -> html.Div:
        """Set up the layout for the page"""
        return html.Div(
            children=[
                html.H2(self.page_name),
                dbc.Row(
                    [
                        dbc.Col([html.H4("Regression Champions"), self.components["reg_champions"]], width=7),
                        dbc.Col([html.H4("Classification Champions"), self.components["class_champions"]], width=5),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col([html.H4("Challengers"), self.components["challengers"]], width=5),
                        dbc.Col(
                            [
                                html.H4("Comparison"),
                                self.components["metrics"],
                                html.Div(self.components["plot"], style={"marginTop": "10px"}),
                            ],
                            width=7,
                        ),
                    ],
                    style={"marginTop": "20px"},
                ),
                # Interval that triggers once on page load
                dcc.Interval(id="mc-page-load", interval=100, max_intervals=1),
            ],
            style={"margin": "30px"},
        )

    def page_load_callbacks(self):
        """Populate the champion tables on page load"""

        @callback(
            [Output(cid, prop) for t in (self.reg_champions, self.class_champions) for cid, prop in t.properties],
            Input("mc-page-load", "n_intervals"),
        )
        def _populate_champions(_n):
            self.view.refresh()
            reg_props = self.reg_champions.update_properties(self.view.reg_champions)
            class_props = self.class_champions.update_properties(self.view.class_champions)
            return reg_props + class_props

    def page_callbacks(self):
        """Selection callbacks: champion -> challengers, challenger -> comparison"""

        @callback(
            [Output(cid, prop) for cid, prop in self.challengers.properties]
            + [
                # Selecting in one champion table clears the other (and any stale challenger pick)
                Output("mc_reg_champions", "selectedRows"),
                Output("mc_class_champions", "selectedRows"),
                Output("mc_challengers", "selectedRows"),
            ],
            Input("mc_reg_champions", "selectedRows"),
            Input("mc_class_champions", "selectedRows"),
            prevent_initial_call=True,
        )
        def _champion_selected(reg_rows, class_rows):
            from_reg = ctx.triggered_id == "mc_reg_champions"
            rows = reg_rows if from_reg else class_rows
            if not rows or rows[0] is None:
                raise PreventUpdate
            challengers_df = self.view.challengers(rows[0]["Endpoint"])
            clear_other = [no_update, []] if from_reg else [[], no_update]
            return self.challengers.update_properties(challengers_df) + clear_other + [[]]

        @callback(
            [Output(cid, prop) for t in (self.metrics, self.plot) for cid, prop in t.properties],
            Input("mc_challengers", "selectedRows"),
            State("mc_reg_champions", "selectedRows"),
            State("mc_class_champions", "selectedRows"),
            prevent_initial_call=True,
        )
        def _challenger_selected(chal_rows, reg_rows, class_rows):
            champ_rows = reg_rows or class_rows
            if not chal_rows or chal_rows[0] is None or not champ_rows:
                raise PreventUpdate
            champion, challenger = champ_rows[0]["Model"], chal_rows[0]["Model"]
            metrics_props = self.metrics.update_properties(self.view.comparison(champion, challenger))

            # A model without predictions for the run (e.g. a fresh champion copy) keeps the old plot
            plot_df = self.view.plot_data(champion, challenger)
            plot_props = (
                self.plot.update_properties(plot_df) if plot_df is not None else [no_update] * len(self.plot.properties)
            )
            return metrics_props + plot_props


# Unit Test for your Plugin Page
if __name__ == "__main__":
    import webbrowser
    from workbench.utils.theme_manager import ThemeManager

    # Set up the Theme Manager
    tm = ThemeManager()
    tm.set_theme("dark")
    css_files = tm.css_files()

    # Create the Dash app
    my_app = dash.Dash(
        __name__, title="Workbench Dashboard", use_pages=True, external_stylesheets=css_files, pages_folder=""
    )
    my_app.layout = html.Div(
        [
            dbc.Container([page_container], fluid=True, className="dbc dbc-ag-grid"),
        ],
        **{"data-bs-theme": tm.data_bs_theme()},
    )

    # Create the Plugin Page and call page_setup
    plugin_page = ModelComparisonPage()
    plugin_page.page_setup(my_app)

    # Open the browser to the plugin page
    webbrowser.open("http://localhost:8000/model_comparison")

    # Note: This 'main' is purely for running/testing locally
    my_app.run(host="localhost", port=8000, debug=True)
