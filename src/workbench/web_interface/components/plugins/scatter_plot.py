import base64
import pandas as pd
from dash import dcc, html, callback, Input, Output, no_update
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.theme_manager import ThemeManager


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

        # Initialize the Theme Manager
        self.theme_manager = ThemeManager()

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
        ]
        self.signals = [(f"{component_id}-graph", "hoverData"), (f"{component_id}-graph", "clickData")]

        # Create the Composite Component
        # - A Graph/ScatterPlot Component
        # - Dropdowns for X, Y, and Color
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
                            - dropdown_columns: The columns to use for the x, y, color options
                            - hover_columns: The columns to show when hovering over a point
                            - suppress_hover_display: Suppress hover display (default: False)
                            - custom_data: Custom data that get passed to hoverData callbacks

        Returns:
            list: A list of updated property values (figure, x options, y options, color options).
        """

        # Get the limit for the number of rows to plot
        limit = kwargs.get("limit", 20000)

        # Check that we got a dataframe for our input data object
        if isinstance(input_data, pd.DataFrame):
            if len(input_data) > limit:
                self.log.warning(f"Input data has {len(input_data)} rows, sampling to {limit} rows.")
                self.df = input_data.sample(n=limit)
            else:
                self.df = input_data
        else:
            raise ValueError("The input data must be a Pandas DataFrame.")

        # AWS Feature Groups will also add these implicit columns, so remove these columns
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        self.df = self.df.drop(columns=aws_cols, errors="ignore")

        # Drop any columns with NaNs
        self.df = self.df.dropna(axis=1, how="any")

        # Set the hover columns and custom data
        self.hover_columns = kwargs.get("hover_columns", self.df.columns.tolist()[:10])
        self.suppress_hover_display = kwargs.get("suppress_hover_display", False)
        self.custom_data = kwargs.get("custom_data", [])

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

        return [figure, options, options, options, x_default, y_default, color_default]

    def create_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
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

        def compute_opacity(value, min_val, max_val):
            """Normalize and compute opacity based on value."""
            return 0.5 + 0.49 * (value - min_val) / (max_val - min_val)

        def generate_hover_text(row):
            """Generate hover text for each data point."""
            return "<br>".join([f"{col}: {row[col]}" for col in self.hover_columns])

        # Cache min and max for color_col
        color_min, color_max = df[color_col].min(), df[color_col].max()

        # Generate hover text for all points
        hovertext = df.apply(generate_hover_text, axis=1)
        hovertemplate = "%{hovertext}<extra></extra>"
        hoverinfo = None

        # Suppress the display of hover info
        if self.suppress_hover_display:
            hoverinfo = "none" if self.suppress_hover_display else None
            hovertemplate = None

        # Create the scatter plot
        figure = go.Figure(
            data=go.Scattergl(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                hoverinfo=hoverinfo,
                hovertext=hovertext,
                hovertemplate=hovertemplate,
                customdata=df[self.custom_data],
                marker=dict(
                    size=marker_size,
                    color=df[color_col],
                    colorscale=self.theme_manager.colorscale(),
                    colorbar=dict(title=color_col, thickness=20),
                    opacity=df[color_col].apply(lambda x: compute_opacity(x, color_min, color_max)),
                    line=dict(color="rgba(0,0,0,1)", width=1),
                ),
            )
        )

        # Add 45-degree line if enabled
        if regression_line:
            axis_min, axis_max = min(df[x_col].min(), df[y_col].min()), max(df[x_col].max(), df[y_col].max())
            figure.add_shape(
                type="line",
                line=dict(width=4, color="rgba(128, 128, 128, 0.5)"),
                x0=axis_min,
                x1=axis_max,
                y0=axis_min,
                y1=axis_max,
            )

        # Logic for axis labels
        if self.show_axes:
            xaxis = dict(
                title=x_col,
                tickformat=".2f",
            )
            yaxis = dict(
                title=y_col,
                tickformat=".2f",
            )
        else:
            xaxis = dict(visible=False)
            yaxis = dict(visible=False)

        # Update layout
        figure.update_layout(
            margin={"t": 30, "b": 40, "r": 30, "l": 70, "pad": 10},
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=False,
            dragmode="pan",
            modebar={"bgcolor": "rgba(0, 0, 0, 0)"},  # Transparent background
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
                Input(f"{self.component_id}-regression-line", "value"),
            ],
            prevent_initial_call=True,
        )
        def _update_scatter_plot(x_value, y_value, color_value, regression_line):
            """Update the Scatter Plot Graph based on the dropdown values."""

            # Check if the dataframe is not empty and the values are not None
            if not self.df.empty and x_value and y_value and color_value:
                # Update Plotly Scatter Plot
                figure = self.create_scatter_plot(self.df, x_value, y_value, color_value, regression_line)
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
    from workbench.api import df_store

    # Load our preprocessed tox21 training data
    df = df_store.DFStore().get("/datasets/chem_info/tox21")

    # Run the Unit Test on the Plugin
    PluginUnitTest(ScatterPlot, input_data=df, theme="dark", suppress_hover_display=True, x="x", y="y").run()
