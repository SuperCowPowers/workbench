"""ExamplePlot: A minimal 'hello world' Workbench plugin.

Use this as a template for writing your own plugins. A plugin needs two things:
  1. Two class attributes: `auto_load_page` and `plugin_input_type`
  2. Two methods: `create_component` (build the UI) and `update_properties` (fill it with data)

Everything else (internal callbacks, theme handling) is optional.
"""

import pandas as pd
from dash import dcc, html
import plotly.graph_objects as go

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType


class ExamplePlot(PluginInterface):
    """A minimal Scatter Plot Plugin that takes a DataFrame and plots its first two numeric columns."""

    # Required class attributes:
    #   auto_load_page: which Workbench page auto-loads this plugin (NONE = don't autoload)
    #   plugin_input_type: the kind of object update_properties() receives
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    def create_component(self, component_id: str) -> html.Div:
        """Build the (data-less) Dash layout for this plugin.

        Called once at startup. Lay out your components here, register which of their
        properties Workbench will update later via `self.properties`, and return the
        top-level component. Do NOT put data in here yet (see update_properties).

        Args:
            component_id (str): Unique ID prefix for this plugin's web components.

        Returns:
            html.Div: The top-level Dash component for this plugin.
        """
        self.component_id = component_id

        # Declare which (component_id, property) pairs Workbench will update. The order
        # here must match the order of the list returned by update_properties().
        self.properties = [(f"{component_id}-graph", "figure")]

        # Return an empty graph; it gets populated by update_properties().
        return html.Div(
            children=[
                dcc.Graph(id=f"{component_id}-graph", figure=self.display_text("Waiting for Data...")),
            ]
        )

    def update_properties(self, input_data: pd.DataFrame, **kwargs) -> list:
        """Fill the plugin's components with data.

        Called whenever new data arrives. Build your figure (or other property values)
        from `input_data` and return them in the SAME ORDER as `self.properties`.

        Args:
            input_data (pd.DataFrame): The input DataFrame to plot.
            **kwargs: Optional plugin-specific arguments. This example supports:
                      - x: Column for the x-axis (defaults to the first numeric column)
                      - y: Column for the y-axis (defaults to the second numeric column)

        Returns:
            list: Updated property values, one per entry in self.properties (here: [figure]).
        """
        # Plot the requested columns, or fall back to the first two numeric columns.
        numeric_columns = input_data.select_dtypes(include="number").columns.tolist()
        x_col = kwargs.get("x", numeric_columns[0])
        y_col = kwargs.get("y", numeric_columns[1])
        figure = go.Figure(data=go.Scatter(x=input_data[x_col], y=input_data[y_col], mode="markers"))

        # Must return a list matching self.properties order.
        return [figure]


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.api import Model

    # Grab a real DataFrame of inference predictions from a Workbench model.
    model = Model("logd-reg-xgb")
    df = model.get_inference_predictions("full_cross_fold")

    # Run the Unit Test on the Plugin (x/y kwargs are forwarded to update_properties).
    PluginUnitTest(ExamplePlot, input_data=df, x="logd", y="prediction").run()
