import base64
import pandas as pd
from dash import dcc, html, callback, Input, Output, no_update
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType


class ScatterPlot(PluginInterface):
    """A Scatter Plot Plugin for Feature Sets."""

    # Initialize this Plugin Component Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    def __init__(self, show_axes: bool = True):
        """Initialize the Scatter Plot Plugin

        Args:
            show_axes (bool): Whether to show the axes and grid. Default is True.
        """
        self.component_id = None
        self.hover_columns = []
        self.df = None
        self.show_axes = show_axes

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
            (f"{component_id}-label-dropdown", "options"),
            (f"{component_id}-x-dropdown", "value"),
            (f"{component_id}-y-dropdown", "value"),
            (f"{component_id}-color-dropdown", "value"),
        ]
        self.signals = [(f"{component_id}-graph", "hoverData"), (f"{component_id}-graph", "clickData")]

        # Create the Composite Component
        # - A Graph/ScatterPlot Component
        # - Dropdowns for X, Y, Color, and Label
        # - Checkbox for Regression Line
        return html.Div(
            className="workbench-container",
            children=[
                # Main Scatter Plot Graph
                dcc.Graph(
                    id=f"{component_id}-graph",
                    figure=self.display_text("Waiting for Data..."),
                    config={"scrollZoom": True},
                    style={"height": "100%"},
                    clear_on_unhover=True,
                ),
                # Controls: X, Y, Color, Label Dropdowns, and Regression Line Checkbox
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
                        html.Label("Label", style={"marginLeft": "30px", "marginRight": "5px", "fontWeight": "bold"}),
                        dcc.Dropdown(
                            id=f"{component_id}-label-dropdown",
                            className="dropdown",
                            style={"min-width": "50px", "flex": 1},
                            options=[{"label": "None", "value": "none"}],
                            value="none",
                            clearable=False,
                        ),
                        dcc.Checklist(
                            id=f"{component_id}-regression-line",
                            options=[{"label": " Diagonal", "value": "show"}],
                            value=[],
                            style={"margin": "10px"},
                        ),
                    ],
                    style={"padding": "0px 0px 10px 0px", "display": "flex", "gap": "10px"},
                ),
                dcc.Tooltip(
                    id=f"{component_id}-overlay",
                    background_color="rgba(0,0,0,0)",
                    border_color="rgba(0,0,0,0)",
                    direction="bottom",
                    loading_text="",
                ),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column"},  # Full viewport height
        )

    def update_properties(self, input_data: pd.DataFrame, **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            input_data (pd.DataFrame): The input data object.
            **kwargs: Additional keyword arguments (plugins can define their own arguments).
                      Note: The current kwargs processed are:
                            - x: The default x-axis column
                            - y: The default y-axis column
                            - color: The default color column
                            - dropdown_columns: The columns to use for the x and y options
                            - hover_columns: The columns to show when hovering over a point
                            - suppress_hover_display: Suppress hover display (default: False)
                            - custom_data: Custom data that get passed to hoverData callbacks

        Returns:
            list: A list of updated property values (figure, x options, y options, color options,
                                                    label options, x default, y default,
                                                    color default).
        """
        # Get the limit for the number of rows to plot
        limit = kwargs.get("limit", 20000)

        # Ensure input_data is a DataFrame and sample if necessary
        if isinstance(input_data, pd.DataFrame):
            self.df = input_data.sample(n=limit) if len(input_data) > limit else input_data
        else:
            raise ValueError("The input data must be a Pandas DataFrame.")

        # Remove AWS created columns
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        self.df = self.df.drop(columns=aws_cols, errors="ignore")

        # Set hover columns and custom data
        self.hover_columns = kwargs.get("hover_columns", self.df.columns.tolist()[:10])
        self.suppress_hover_display = kwargs.get("suppress_hover_display", False)
        self.custom_data = kwargs.get("custom_data", [])

        # Identify numeric columns
        numeric_columns = self.df.select_dtypes(include="number").columns.tolist()
        if len(numeric_columns) < 3:
            raise ValueError("At least three numeric columns are required for x, y, and color.")

        # Default x, y, and color (for color, default to a numeric column)
        x_default = kwargs.get("x", numeric_columns[0])
        y_default = kwargs.get("y", numeric_columns[1])
        color_default = kwargs.get("color", numeric_columns[2])
        regression_line = kwargs.get("regression_line", False)

        # Create the default scatter plot
        figure = self.create_scatter_plot(self.df, x_default, y_default, color_default, "none", regression_line)

        # Dropdown options for x and y: use provided dropdown_columns or fallback to numeric columns
        dropdown_columns = kwargs.get("dropdown_columns", numeric_columns)
        x_options = [{"label": col, "value": col} for col in dropdown_columns]
        y_options = x_options.copy()

        # For color dropdown include any categorical columns (with less than 20 unique values)
        cat_columns = self.df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        cat_columns = [col for col in cat_columns if self.df[col].astype(str).nunique() < 20]
        color_columns = numeric_columns + cat_columns
        color_options = [{"label": col, "value": col} for col in color_columns]

        # For label dropdown, include None option and all columns
        label_options = [{"label": "None", "value": "none"}]
        label_options.extend([{"label": col, "value": col} for col in self.df.columns])

        return [figure, x_options, y_options, color_options, label_options, x_default, y_default, color_default]

    def create_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        label_col: str,
        regression_line: bool = False,
        marker_size: int = 15,
    ) -> go.Figure:
        """Create a Plotly Scatter Plot figure.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            x_col (str): The column to use for the x-axis.
            y_col (str): The column to use for the y-axis.
            color_col (str): The column to use for the color scale.
            regression_line (bool): Whether to include a regression line.
            marker_size (int): Size of the markers. Default is 15.

        Returns:
            go.Figure: A Plotly Figure object.
        """
        # Check if we need to show labels
        show_labels = label_col != "none" and len(df) < 1000

        # Helper to generate hover text for each point.
        def generate_hover_text(row):
            return "<br>".join([f"{col}: {row[col]}" for col in self.hover_columns])

        # Generate hover text for all points.
        hovertext = df.apply(generate_hover_text, axis=1)
        hovertemplate = "%{hovertext}<extra></extra>"
        hoverinfo = "none" if self.suppress_hover_display else None

        # Determine marker settings based on the type of the color column.
        if pd.api.types.is_numeric_dtype(df[color_col]):
            marker_color = df[color_col]
            colorbar = dict(title=color_col, thickness=20)
            # Single trace for numeric data.
            data = [
                go.Scattergl(
                    x=df[x_col],
                    y=df[y_col],
                    mode="markers+text" if show_labels else "markers",
                    text=df[label_col].astype(str) if show_labels else None,
                    textposition="top center",
                    hoverinfo=hoverinfo,
                    hovertext=hovertext,
                    hovertemplate=hovertemplate,
                    customdata=df[self.custom_data],
                    marker=dict(
                        size=marker_size,
                        color=marker_color,
                        colorbar=colorbar,
                        opacity=0.8,
                        line=dict(color="rgba(0,0,0,0.25)", width=1),
                    ),
                )
            ]
            showlegend = False
        else:
            # For categorical data, create one trace per category so that a legend appears.
            categories = sorted(df[color_col].astype(str).unique())
            # Use the provided colorscale as a discrete palette.
            # Hardcode a discrete colorscale using Plotly Express's qualitative palette.
            import plotly.express as px

            discrete_colors = px.colors.qualitative.Plotly
            data = []
            for i, cat in enumerate(categories):
                sub_df = df[df[color_col] == cat]
                sub_hovertext = hovertext.loc[sub_df.index]
                trace = go.Scattergl(
                    x=sub_df[x_col],
                    y=sub_df[y_col],
                    mode="markers+text" if show_labels else "markers",  # Add text mode if labels enabled
                    text=sub_df[label_col] if show_labels else None,  # Add text if labels enabled
                    textposition="top center",  # Position labels above points
                    name=cat,
                    hoverinfo=hoverinfo,
                    hovertext=sub_hovertext,
                    hovertemplate=hovertemplate,
                    customdata=sub_df[self.custom_data],
                    marker=dict(
                        size=marker_size,
                        color=discrete_colors[i % len(discrete_colors)],
                        opacity=0.8,
                        line=dict(color="rgba(0,0,0,0.25)", width=1),
                    ),
                )
                data.append(trace)
            showlegend = True

        figure = go.Figure(data=data)

        # Add regression line if enabled.
        if regression_line:
            axis_min = min(df[x_col].min(), df[y_col].min())
            axis_max = max(df[x_col].max(), df[y_col].max())
            figure.add_shape(
                type="line",
                line=dict(width=4, color="rgba(128, 128, 128, 0.5)"),
                x0=axis_min,
                x1=axis_max,
                y0=axis_min,
                y1=axis_max,
            )

        # Set up axes.
        if self.show_axes:
            xaxis = dict(title=x_col, tickformat=".2f")
            yaxis = dict(title=y_col, tickformat=".2f")
        else:
            xaxis = dict(visible=False)
            yaxis = dict(visible=False)

        # Update layout.
        figure.update_layout(
            margin={"t": 30, "b": 40, "r": 30, "l": 70, "pad": 10},
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=showlegend,
            dragmode="pan",
            modebar={"bgcolor": "rgba(0, 0, 0, 0)"},
            uirevision="constant",
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
                Input(f"{self.component_id}-label-dropdown", "value"),
                Input(f"{self.component_id}-regression-line", "value"),
            ],
            prevent_initial_call=True,
        )
        def _update_scatter_plot(x_value, y_value, color_value, label_value, regression_line):
            """Update the Scatter Plot Graph based on the dropdown values."""

            # Check if the dataframe is not empty and the values are not None
            if not self.df.empty and x_value and y_value and color_value:
                # Update Plotly Scatter Plot with the label value
                figure = self.create_scatter_plot(self.df, x_value, y_value, color_value, label_value, regression_line)
                return figure

            raise PreventUpdate

        @callback(
            Output(f"{self.component_id}-overlay", "show"),
            Output(f"{self.component_id}-overlay", "bbox"),
            Output(f"{self.component_id}-overlay", "children"),
            Input(f"{self.component_id}-graph", "hoverData"),
        )
        def _scatter_overlay(hover_data):
            if hover_data is None:
                # Hide the overlay if no hover data
                return False, no_update, no_update

            # Extract bounding box from hoverData
            bbox = hover_data["points"][0]["bbox"]

            # Create an SVG with a circle at the center
            svg = """
            <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" style="overflow: visible;">
                <!-- Circle for the node -->
                <circle cx="50" cy="50" r="10" stroke="rgba(255, 255, 255, 1)" stroke-width="3" fill="none" />
            </svg>
            """

            # Encode the SVG as Base64
            encoded_svg = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
            data_uri = f"data:image/svg+xml;base64,{encoded_svg}"

            # Use an img tag for the overlay
            svg_image = html.Img(src=data_uri, style={"width": "100px", "height": "100px"})

            # Get the center of the bounding box
            center_x = (bbox["x0"] + bbox["x1"]) / 2
            center_y = (bbox["y0"] + bbox["y1"]) / 2

            # The tooltip should be centered on the point (note: 'bottom' tooltip, so we adjust y position)
            adjusted_bbox = {
                "x0": center_x - 50,
                "x1": center_x + 50,
                "y0": center_y - 162,
                "y1": center_y - 62,
            }
            # Return the updated values for the overlay
            return True, adjusted_bbox, [svg_image]


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Create a fake dataframe with 3 numeric columns and 2 categorical columns
    data = {
        "x": [1, 2, 3, 4, 5],
        "y": [2, 3, 4, 5, 6],
        "z": [3, 4, 5, 6, 7],
        "class": ["A", "C", "B", "B", "A"],
        "label": ["good", "bad", "okay", "good", "bad"],
    }
    df = pd.DataFrame(data)

    # Run the Unit Test on the Plugin
    PluginUnitTest(ScatterPlot, input_data=df, theme="dark", suppress_hover_display=True).run()
