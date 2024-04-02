"""MyPluginPage: An Example SageWorks Plugin Page"""

import dash
from dash import register_page, dcc
import dash_bootstrap_components as dbc
import logging
import plotly.graph_objs as go

# SageWorks Imports
from sageworks.utils.plugin_manager import PluginManager


class MyPluginPage:
    """MyPluginPage:  A SageWorks Example Plugin Page"""

    def __init__(self):
        """Initialize the Plugin Page"""
        self.log = logging.getLogger("sageworks")
        self.app = None
        self.graph_1 = None
        self.graph_2 = None

        # Get our view from the PluginManager
        pm = PluginManager()
        self.my_view = pm.get_view("MyViewPlugin")

    def page_setup(self, app: dash.Dash):
        """Required function to set up the page"""

        # Save the app for later
        self.app = app

        # Some simple Dash/plotly components
        self.graph_1 = dcc.Graph(id="graph-1", figure={"data": [go.Pie(labels=["A", "B", "C"], values=[10, 20, 30])]})
        self.graph_2 = dcc.Graph(id="graph-2", figure={"data": [go.Bar(x=["a", "b", "c"], y=[4, 2, 3])]})

        # Register this page with Dash and set up the layout
        register_page(
            __name__,
            path="/plugin",
            name="My Plugin Page",
            layout=self.page_layout(),
        )

        # Setup Callbacks
        self.setup_callbacks()

    def page_layout(self) -> dash.html.Div:
        """Set up the layout for the page"""
        layout = dash.html.Div(
            children=[
                dbc.Row(
                    [
                        dash.html.H2("My Plugin Page"),
                        dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                    ]
                ),
                # Some Rows for my graphs/plots
                dbc.Row(self.graph_1),
                dbc.Row(self.graph_2),
            ],
            style={"margin": "30px"},
        )
        return layout

    def setup_callbacks(self):
        """Set up the callbacks for the page"""
        pass


if __name__ == "__main__":
    # Create a Dash app for testing
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    # Instantiate the plugin page
    plugin_page = MyPluginPage()

    # Call page setup
    plugin_page.page_setup(app)

    # Manually set the layout of the app to the layout of the plugin page
    app.layout = plugin_page.page_layout()

    # Run the app in debug mode
    app.run(debug=True)
