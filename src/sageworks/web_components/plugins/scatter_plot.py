from typing import Union
import pandas as pd
from dash import dcc, html, callback, Input, Output
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc  # Import Dash Bootstrap Components

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType


class ScatterPlot(PluginInterface):
    """A Scatter Plot Plugin for Feature Sets."""

    # Initialize this Plugin Component Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.FEATURE_SET

    def __init__(self, theme: str = "LIGHT"):
        """Initialize the Scatter Plot Plugin.

        Args:
            theme (str): The theme to use for the plot. Default is "DARKLY".
        """
        self.component_id = None
        self.hover_columns = []
        self.df = None

        # Store the theme for the plot
        if theme == "DARKLY":
            self.theme = dbc.themes.DARKLY
            self.dark_theme = True
        else:
            self.theme = dbc.themes.BOOTSTRAP
            self.dark_theme = False

        # Call the parent class constructor
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Dash Graph Component without any data.

        Args:
            component_id (str): The ID of the web component.

        Returns:
            html.Div: A Dash Div Component containing the graph and dropdowns.
        """
        self.component_id = component_id

        # Determine if the theme is dark
        is_dark_theme = self.theme == dbc.themes.DARKLY

        # Styles for components based on the theme
        label_style = {"color": "white" if is_dark_theme else "black", "marginRight": "5px"}
        dropdown_style = {
            "flex": "1",
            "backgroundColor": "#333" if is_dark_theme else "white",
            "color": "white" if is_dark_theme else "black",
            "border": "1px solid #444" if is_dark_theme else "1px solid #ccc",
        }
        checklist_style = {"flex": "1", "color": "white" if is_dark_theme else "black"}

        # Fill in plugin properties and signals
        self.properties = [
            (f"{component_id}-graph", "figure"),
            (f"{component_id}-x-dropdown", "options"),
            (f"{component_id}-y-dropdown", "options"),
            (f"{component_id}-color-dropdown", "options"),
            (f"{component_id}-x-dropdown", "value"),
            (f"{component_id}-y-dropdown", "value"),
            (f"{component_id}-color-dropdown", "value"),
            (f"{component_id}-regression-line", "value"),
        ]
        self.signals = [(f"{component_id}-graph", "hoverData")]

        return html.Div(
            [
                dcc.Graph(
                    id=f"{component_id}-graph",
                    figure=self.display_text("Waiting for Data..."),
                    config={"scrollZoom": True},
                ),
                html.Div(
                    [
                        html.Label("X", style=label_style),
                        dcc.Dropdown(
                            id=f"{component_id}-x-dropdown",
                            placeholder="Select X-axis",
                            value=None,
                            style=dropdown_style,
                            className="custom-dropdown" if is_dark_theme else "",
                        ),
                        html.Label("Y", style={**label_style, "marginLeft": "20px"}),
                        dcc.Dropdown(
                            id=f"{component_id}-y-dropdown",
                            placeholder="Select Y-axis",
                            value=None,
                            style=dropdown_style,
                            className="custom-dropdown" if is_dark_theme else "",
                        ),
                        html.Label("Color", style={**label_style, "marginLeft": "20px"}),
                        dcc.Dropdown(
                            id=f"{component_id}-color-dropdown",
                            placeholder="Select Color",
                            value=None,
                            style=dropdown_style,
                            className="custom-dropdown" if is_dark_theme else "",
                        ),
                        dcc.Checklist(
                            id=f"{component_id}-regression-line",
                            options=[{"label": "Regression Line", "value": "show"}],
                            value=[],
                            style=checklist_style,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "padding": "10px 0",
                    },
                ),
            ]
        )

    def update_properties(self, input_data: Union[DataSource, FeatureSet, pd.DataFrame], **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            input_data (DataSource or FeatureSet): The input data object.
            **kwargs: Additional keyword arguments (plugins can define their own arguments).
                      Note: The current argument processed are:
                            - x: The default x-axis column
                            - y: The default y-axis column
                            - color: The default color column
                            - dropdown_columns: The columns to use for the x, y, color options
                            - hover_columns: The columns to show when hovering over a point

        Returns:
            list: A list of updated property values (figure, x options, y options, color options).
        """

        # Grab the dataframe from the input data object
        if isinstance(input_data, (DataSource, FeatureSet)):
            self.df = input_data.view("display").pull_dataframe(limit=1000)
        else:
            self.df = input_data

        # Set the default hover columns
        self.hover_columns = kwargs.get("hover_columns", self.df.columns.tolist()[:10])

        # AWS Feature Groups will also add these implicit columns, so remove these columns
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time", "training"]
        self.df = self.df.drop(columns=aws_cols, errors="ignore")

        # Get numeric columns for default selections
        numeric_columns = self.df.select_dtypes(include="number").columns.tolist()
        if len(numeric_columns) < 3:
            raise ValueError("At least three numeric columns are required for x, y, and color.")

        # Check if the kwargs are provided for x, y, and color
        x_default = kwargs.get("x", numeric_columns[0])
        y_default = kwargs.get("y", numeric_columns[1])
        color_default = kwargs.get("color", numeric_columns[2])

        # Create default Plotly Scatter Plot
        figure = self.create_scatter_plot(self.df, x_default, y_default, color_default)

        # Dropdown options for x, y, and color
        if "dropdown_columns" in kwargs:
            dropdown_columns = kwargs["dropdown_columns"]
            dropdown_columns = [col for col in dropdown_columns if col in numeric_columns]
        else:
            dropdown_columns = numeric_columns
        options = [{"label": col, "value": col} for col in dropdown_columns]

        return [figure, options, options, options, x_default, y_default, color_default, []]

    def create_scatter_plot(self, df, x_col, y_col, color_col, regression_line=False):
        """Create a Plotly Scatter Plot figure.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            x_col (str): The column to use for the x-axis.
            y_col (str): The column to use for the y-axis.
            color_col (str): The column to use for the color scale.
            regression_line (bool): Whether to include a regression line.

        Returns:
            go.Figure: A Plotly Figure object.
        """

        # Put in a hard limit on the number of rows to plot
        if len(df) > 10000:
            print(f"Too many rows to plot ({len(df)}), limiting to 10000 rows.")
        df = df.head(10000)

        # Define a custom color scale (blue -> yellow -> orange -> red)
        color_scale = [
            [0.0, "rgb(64,64,160)"],
            [0.33, "rgb(48, 140, 140)"],
            [0.67, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"],
        ]

        # Create Plotly Scatter Plot
        figure = go.Figure(
            data=go.Scattergl(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                hovertext=df.apply(
                    lambda row: "<br>".join([f"{col}: {row[col]}" for col in self.hover_columns]), axis=1
                ),
                hovertemplate="%{hovertext}<extra></extra>",  # Define hover template and remove extra info
                textfont=dict(family="Arial Black", size=14),  # Set font size
                marker=dict(
                    size=15,
                    color=df[color_col],  # Use the selected field for color
                    colorscale=color_scale,
                    colorbar=dict(title=color_col),
                    opacity=df[color_col].apply(
                        lambda x: 0.25 + 0.74 * (x - df[color_col].min()) / (df[color_col].max() - df[color_col].min())
                    ),
                    line=dict(color="Black", width=1),
                ),
            )
        )

        # Add 45-degree line
        if regression_line:
            min_val = min(df[x_col].min(), df[y_col].min())
            max_val = max(df[x_col].max(), df[y_col].max())
            figure.add_shape(
                type="line",
                line=dict(width=5, color="rgba(1.0, 1.0, 1.0, 0.5)"),
                x0=min_val,
                x1=max_val,
                y0=min_val,
                y1=max_val,
            )

        # Apply the selected theme
        plotly_theme = "plotly_dark" if self.dark_theme else "plotly"
        figure.update_layout(
            template=plotly_theme,
            margin={"t": 40, "b": 40, "r": 40, "l": 40, "pad": 0},
            xaxis=dict(title=x_col, tickformat=".2f"),  # Add x-axis title
            yaxis=dict(title=y_col, tickformat=".2f"),  # Add y-axis title
            showlegend=False,  # Remove legend
            dragmode="pan",
        )
        return figure

    def register_internal_callbacks(self):
        """Register any internal callbacks for the plugin."""

        @callback(
            Output(f"{self.component_id}-graph", "figure", allow_duplicate=True),
            [
                Input(f"{self.component_id}-x-dropdown", "value"),
                Input(f"{self.component_id}-y-dropdown", "value"),
                Input(f"{self.component_id}-color-dropdown", "value"),
                Input(f"{self.component_id}-regression-line", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_graph(x_value, y_value, color_value, regression_line):
            # Get the latest dataframe
            df = self.df

            if not df.empty and x_value and y_value and color_value:
                # Update Plotly Scatter Plot
                figure = self.create_scatter_plot(df, x_value, y_value, color_value, regression_line)
                return figure

            raise PreventUpdate


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(ScatterPlot).run()
