from typing import Union
import pandas as pd
from dash import dcc, html, callback, Input, Output
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType


class ScatterPlot(PluginInterface):
    """A Scatter Plot Plugin for Feature Sets."""

    # Initialize this Plugin Component Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.FEATURE_SET

    def __init__(self):
        """Initialize the Scatter Plot Plugin"""
        self.component_id = None
        self.hover_columns = []
        self.df = None

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

        # Create the Composite Component
        # - A Graph/ScatterPlot Component
        # - Dropdowns for X, Y, and Color
        # - Checkbox for Regression Line
        return html.Div(
            [
                # Main Scatter Plot Graph
                dcc.Graph(
                    id=f"{component_id}-graph",
                    figure=self.display_text("Waiting for Data..."),
                    config={"scrollZoom": True},
                    style={"width": "100%", "height": "100%"},  # Let the graph fill its container
                ),
                # Controls: X, Y, Color Dropdowns, and Regression Line Checkbox
                html.Div(
                    [
                        html.Label("X", style={"marginLeft": "40px", "marginRight": "5px", "fontWeight": "bold"}),
                        dcc.Dropdown(
                            id=f"{component_id}-x-dropdown",
                            className="dropdown",
                            style={"min-width": "50px", "flex": 1},  # Responsive width
                            clearable=False,
                        ),
                        html.Label("Y", style={"marginLeft": "30px", "marginRight": "5px", "fontWeight": "bold"}),
                        dcc.Dropdown(
                            id=f"{component_id}-y-dropdown",
                            className="dropdown",
                            style={"min-width": "50px", "flex": 1},  # Responsive width
                            clearable=False,
                        ),
                        html.Label("Color", style={"marginLeft": "30px", "marginRight": "5px", "fontWeight": "bold"}),
                        dcc.Dropdown(
                            id=f"{component_id}-color-dropdown",
                            className="dropdown",
                            style={"min-width": "50px", "flex": 1},  # Responsive width
                            clearable=False,
                        ),
                        dcc.Checklist(
                            id=f"{component_id}-regression-line",
                            options=[{"label": " Diagonal", "value": "show"}],
                            value=[],
                            style={"margin": "10px"},
                        ),
                    ],
                    style={"padding": "10px", "display": "flex", "gap": "10px"},
                ),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column"},  # Full viewport height
        )

    def update_properties(self, input_data: Union[DataSource, FeatureSet, pd.DataFrame], **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            input_data (DataSource or FeatureSet or Pandas dataframe): The input data object.
            **kwargs: Additional keyword arguments (plugins can define their own arguments).
                      Note: The current kwargs processed are:
                            - x: The default x-axis column
                            - y: The default y-axis column
                            - color: The default color column
                            - dropdown_columns: The columns to use for the x, y, color options
                            - hover_columns: The columns to show when hovering over a point

        Returns:
            list: A list of updated property values (figure, x options, y options, color options).
        """

        # Get the limit for the number of rows to plot
        limit = kwargs.get("limit", 10000)

        # Grab the dataframe from the input data object
        if isinstance(input_data, (DataSource, FeatureSet)):
            self.df = input_data.view("display").pull_dataframe(limit=limit)
        elif isinstance(input_data, pd.DataFrame):
            if len(input_data) > limit:
                self.log.warning(f"Input data has {len(input_data)} rows, sampling to {limit} rows.")
                self.df = input_data.sample(n=limit)
            else:
                self.df = input_data
        else:
            raise ValueError("The input data must be a DataSource, FeatureSet, or Pandas DataFrame.")

        # AWS Feature Groups will also add these implicit columns, so remove these columns
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        self.df = self.df.drop(columns=aws_cols, errors="ignore")

        # Drop any columns with NaNs
        self.df = self.df.dropna(axis=1, how="any")

        # Set the default hover columns
        self.hover_columns = kwargs.get("hover_columns", self.df.columns.tolist()[:10])

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

        # Define a custom color scale (blue -> yellow -> orange -> red)
        color_scale = [
            [0.0, "rgb(64, 64, 160)"],
            [0.33, "rgb(48, 140, 140)"],
            [0.67, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"],
        ]

        # Create an OpenGL Scatter Plot
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
                line=dict(width=4, color="rgba(1.0, 1.0, 1.0, 0.25)"),
                x0=min_val,
                x1=max_val,
                y0=min_val,
                y1=max_val,
            )

        # Apply the selected theme and set transparent background
        plotly_theme = "plotly_dark" if self.dark_theme else "plotly"
        figure.update_layout(
            template=plotly_theme,
            margin={"t": 40, "b": 40, "r": 40, "l": 40, "pad": 0},
            xaxis=dict(
                title=x_col,
                tickformat=".2f",
                showgrid=True,
                gridcolor="rgba(100,100,100,0.25)",  # Medium grey
                zerolinecolor="rgba(80, 80, 150, 0.5)",  # Blue color for the zero line
            ),
            yaxis=dict(
                title=y_col,
                tickformat=".2f",
                showgrid=True,
                gridcolor="rgba(100,100,100,0.25)",  # Medium grey
                zerolinecolor="rgba(80, 80, 150, 0.5)",  # Blue color for the zero line
            ),
            showlegend=False,  # Remove legend
            dragmode="pan",
            paper_bgcolor="rgba(0,0,0,0)",  # Set the paper background to transparent
            plot_bgcolor="rgba(0,0,0,0)",  # Set the plot background to transparent
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
            """Update the Scatter Plot Graph based on the dropdown values."""

            # Check if the dataframe is not empty and the values are not None
            if not self.df.empty and x_value and y_value and color_value:
                # Update Plotly Scatter Plot
                figure = self.create_scatter_plot(self.df, x_value, y_value, color_value, regression_line)
                return figure

            raise PreventUpdate


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(ScatterPlot).run()
