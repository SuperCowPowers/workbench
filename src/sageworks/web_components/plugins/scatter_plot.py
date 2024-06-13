from typing import Union
import pandas as pd
from dash import dcc, html, callback, Input, Output
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType


class ScatterPlot(PluginInterface):
    """A Graph Plot Plugin for NetworkX Graphs."""

    # Initialize this Plugin Component Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.FEATURE_SET

    def create_component(self, component_id: str) -> html.Div:
        """Create a Dash Graph Component without any data.

        Args:
            component_id (str): The ID of the web component.

        Returns:
            html.Div: A Dash Div Component containing the graph and dropdowns.
        """
        self.component_id = component_id
        self.df = None

        # Fill in plugin properties and signals
        self.properties = [
            (f"{component_id}-graph", "figure"),
            (f"{component_id}-x-dropdown", "options"),
            (f"{component_id}-y-dropdown", "options"),
            (f"{component_id}-color-dropdown", "options"),
            (f"{component_id}-x-dropdown", "value"),
            (f"{component_id}-y-dropdown", "value"),
            (f"{component_id}-color-dropdown", "value")
        ]
        self.signals = [(f"{component_id}-graph", "hoverData")]

        return html.Div([
            dcc.Graph(id=f"{component_id}-graph", figure=self.display_text("Waiting for Data...")),
            html.Div([
                html.Label("X", style={'marginRight': '5px'}),
                dcc.Dropdown(id=f"{component_id}-x-dropdown", placeholder="Select X-axis", value=None, style={'flex': '1'}),
                html.Label("Y", style={'marginLeft': '20px', 'marginRight': '5px'}),
                dcc.Dropdown(id=f"{component_id}-y-dropdown", placeholder="Select Y-axis", value=None, style={'flex': '1'}),
                html.Label("Color", style={'marginLeft': '20px', 'marginRight': '5px'}),
                dcc.Dropdown(id=f"{component_id}-color-dropdown", placeholder="Select Color", value=None, style={'flex': '1'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'justifyContent': 'space-between',
                      'padding': '10px 0'})
        ])

    def update_properties(self, input_data: Union[DataSource, FeatureSet, pd.DataFrame], **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            input_data (DataSource or FeatureSet): The input data object.
            **kwargs: Additional keyword arguments (plugins can define their own arguments).

        Returns:
            list: A list of updated property values (figure, x options, y options, color options).
        """

        # Grab the dataframe from the input data object
        if isinstance(input_data, (DataSource, FeatureSet)):
            self.df = input_data.pull_dataframe()
        else:
            self.df = input_data

        # AWS Feature Groups will also add these implicit columns, so remove these columns
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time", "training"]
        self.df = self.df.drop(columns=aws_cols, errors="ignore")

        # Get numeric columns for default selections
        numeric_columns = self.df.select_dtypes(include="number").columns.tolist()
        if len(numeric_columns) < 3:
            raise ValueError("At least three numeric columns are required for x, y, and color.")

        # Default selections for x, y, and color
        x_default = numeric_columns[0]
        y_default = numeric_columns[1]
        color_default = numeric_columns[2]

        # Create default Plotly Scatter Plot
        figure = self.create_scatter_plot(self.df, x_default, y_default, color_default)

        # Dropdown options (just numeric columns)
        options = [{"label": col, "value": col} for col in numeric_columns]

        return [figure, options, options, options, x_default, y_default, color_default]

    @staticmethod
    def create_scatter_plot(df, x_col, y_col, color_col):
        """Create a Plotly Scatter Plot figure.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            x_col (str): The column to use for the x-axis.
            y_col (str): The column to use for the y-axis.
            color_col (str): The column to use for the color scale.

        Returns:
            go.Figure: A Plotly Figure object.
        """
        # Define a custom color scale (blue -> yellow -> orange -> red)
        color_scale = [
            [0.0, "rgb(64,64,160)"],
            [0.33, "rgb(48, 140, 140)"],
            [0.67, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"],
        ]
        # Create Plotly Scatter Plot
        figure = go.Figure(data=go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            hovertext=df.apply(lambda row: "<br>".join([f"{col}: {row[col]}" for col in df.columns]), axis=1),
            hovertemplate="%{hovertext}<extra></extra>",  # Define hover template and remove extra info
            textfont=dict(family="Arial Black", size=14),  # Set font size
            marker=dict(
                size=20,
                color=df[color_col],  # Use the selected field for color
                colorscale=color_scale,
                colorbar=dict(title=color_col),
                line=dict(color="Black", width=1),
            ),
        ))

        # Just some fine-tuning of the plot
        figure.update_layout(
            margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10},
            height=400,
            xaxis=dict(title=x_col),  # Add x-axis title
            yaxis=dict(title=y_col),  # Add y-axis title
            showlegend=False,  # Remove legend
        )

        return figure

    def register_internal_callbacks(self):
        """Register any internal callbacks for the plugin."""

        @callback(
            Output(f"{self.component_id}-graph", "figure", allow_duplicate=True),
            [Input(f"{self.component_id}-x-dropdown", "value"),
             Input(f"{self.component_id}-y-dropdown", "value"),
             Input(f"{self.component_id}-color-dropdown", "value")],
            prevent_initial_call=True
        )
        def update_graph(x_value, y_value, color_value):
            # Get the latest dataframe
            df = self.df

            if not df.empty and x_value and y_value and color_value:
                # Update Plotly Scatter Plot
                figure = self.create_scatter_plot(df, x_value, y_value, color_value)
                return figure

            raise PreventUpdate


if __name__ == "__main__":
    # This class takes in graph details and generates a Graph Plot (go.Figure)
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(ScatterPlot).run()
