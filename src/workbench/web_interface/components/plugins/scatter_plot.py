import base64
import numpy as np
import pandas as pd
from dash import dcc, html, callback, clientside_callback, Input, Output, no_update
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.plot_utils import prediction_intervals
from workbench.utils.chem_utils.vis import molecule_hover_tooltip
from workbench.utils.clientside_callbacks import circle_overlay_callback


class ScatterPlot(PluginInterface):
    """A Scatter Plot Plugin for Feature Sets."""

    # Initialize this Plugin Component Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.DATAFRAME

    # Pre-computed circle overlay SVG
    _circle_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" style="overflow: visible;">
        <circle cx="50" cy="50" r="10" stroke="rgba(255, 255, 255, 1)" stroke-width="3" fill="none" />
    </svg>"""
    _circle_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(_circle_svg.encode('utf-8')).decode('utf-8')}"

    def __init__(self, show_axes: bool = True):
        """Initialize the Scatter Plot Plugin

        Args:
            show_axes (bool): Whether to show the axes and grid. Default is True.
        """
        self.component_id = None
        self.hover_columns = []
        self.df = None
        self.show_axes = show_axes
        self.has_smiles = False  # Track if dataframe has smiles column for molecule hover
        self.smiles_column = None
        self.id_column = None
        self.hover_background = None  # Cached background color for molecule hover tooltip

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
        self.signals = [(f"{component_id}-graph", "hoverData"), (f"{component_id}-graph", "clickData")]

        # Create the Composite Component
        # - A Graph/ScatterPlot Component
        # - Dropdowns for X, Y, Color, and Label
        # - Checkbox for Regression Line
        return html.Div(
            children=[
                # Main Scatter Plot Graph
                dcc.Graph(
                    id=f"{component_id}-graph",
                    figure=self.display_text("Waiting for Data..."),
                    config={"scrollZoom": True},
                    style={"height": "500px", "width": "100%"},
                    clear_on_unhover=True,
                ),
                # Controls: X, Y, Color, Label Dropdowns, and Regression Line Checkbox
                html.Div(
                    [
                        html.Label(
                            "X",
                            style={
                                "marginLeft": "20px",
                                "marginRight": "5px",
                                "fontWeight": "bold",
                                "display": "flex",
                                "alignItems": "center",
                            },
                        ),
                        dcc.Dropdown(
                            id=f"{component_id}-x-dropdown",
                            style={"minWidth": "150px", "flex": 1},
                            clearable=False,
                        ),
                        html.Label(
                            "Y",
                            style={
                                "marginLeft": "20px",
                                "marginRight": "5px",
                                "fontWeight": "bold",
                                "display": "flex",
                                "alignItems": "center",
                            },
                        ),
                        dcc.Dropdown(
                            id=f"{component_id}-y-dropdown",
                            style={"minWidth": "150px", "flex": 1},
                            clearable=False,
                        ),
                        html.Label(
                            "Color",
                            style={
                                "marginLeft": "20px",
                                "marginRight": "5px",
                                "fontWeight": "bold",
                                "display": "flex",
                                "alignItems": "center",
                            },
                        ),
                        dcc.Dropdown(
                            id=f"{component_id}-color-dropdown",
                            style={"minWidth": "150px", "flex": 1},
                            clearable=False,
                        ),
                        dcc.Checklist(
                            id=f"{component_id}-regression-line",
                            options=[{"label": " Diagonal", "value": "show"}],
                            value=[],
                            style={"marginLeft": "20px", "display": "flex", "alignItems": "center"},
                        ),
                    ],
                    style={"padding": "0px 0px 10px 0px", "display": "flex", "alignItems": "center", "gap": "5px"},
                ),
                # Circle overlay tooltip (centered on hovered point)
                dcc.Tooltip(
                    id=f"{component_id}-overlay",
                    background_color="rgba(0,0,0,0)",
                    border_color="rgba(0,0,0,0)",
                    direction="bottom",
                    loading_text="",
                ),
                # Molecule tooltip (offset from hovered point) - only used when smiles column exists
                dcc.Tooltip(
                    id=f"{component_id}-molecule-tooltip",
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
                            - id_column: Column to use for molecule tooltip header (auto-detects "id" if not specified)

        Returns:
            list: A list of updated property values (figure, x options, y options, color options,
                                                    x default, y default, color default).
        """
        # Get the colorscale and background color from the current theme
        self.colorscale = self.theme_manager.colorscale()
        self.hover_background = self.theme_manager.background()

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
        self.hover_columns = kwargs.get("hover_columns", sorted(self.df.columns.tolist()[:15]))
        self.suppress_hover_display = kwargs.get("suppress_hover_display", False)
        self.custom_data = kwargs.get("custom_data", [])

        # Check if the dataframe has smiles/id columns for molecule hover rendering
        self.smiles_column = next((col for col in self.df.columns if col.lower() == "smiles"), None)
        # Use provided id_column, or auto-detect "id" column, or fall back to first column
        self.id_column = kwargs.get("id_column") or next(
            (col for col in self.df.columns if col.lower() == "id"), self.df.columns[0]
        )
        self.has_smiles = self.smiles_column is not None

        # Identify numeric columns
        numeric_columns = self.df.select_dtypes(include="number").columns.tolist()
        if len(numeric_columns) < 3:
            raise ValueError("At least three numeric columns are required for x, y, and color.")

        # Default x, y, and color (for color, prefer 'confidence' if it exists)
        x_default = kwargs.get("x", numeric_columns[0])
        y_default = kwargs.get("y", numeric_columns[1])
        default_color = "confidence" if "confidence" in self.df.columns else numeric_columns[2]
        color_default = kwargs.get("color", default_color)
        regression_line = kwargs.get("regression_line", False)

        # Create the default scatter plot
        figure = self.create_scatter_plot(self.df, x_default, y_default, color_default, regression_line)

        # Dropdown options for x and y: use provided dropdown_columns or fallback to numeric columns
        dropdown_columns = kwargs.get("dropdown_columns", numeric_columns)
        x_options = [{"label": col, "value": col} for col in dropdown_columns]
        y_options = x_options.copy()

        # For color dropdown include any categorical columns (with less than 20 unique values)
        cat_columns = self.df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        cat_columns = [col for col in cat_columns if self.df[col].astype(str).nunique() < 20]
        color_columns = numeric_columns + cat_columns
        color_options = [{"label": col, "value": col} for col in color_columns]

        # Regression line checklist value (list with "show" if enabled, empty list if disabled)
        regression_line_value = ["show"] if regression_line else []

        return [figure, x_options, y_options, color_options, x_default, y_default, color_default, regression_line_value]

    def set_theme(self, theme: str) -> list:
        """Re-render the scatter plot when the theme changes."""
        # If no data yet, return no_update for all properties
        if self.df is None or self.df.empty:
            return [no_update] * len(self.properties)

        # Re-render with defaults (user dropdown selections reset, but theme changes are rare)
        return self.update_properties(self.df)

    def create_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        regression_line: bool = False,
    ) -> go.Figure:
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

        # If aggregation_count is present, sort so largest counts are drawn first (underneath)
        # and compute marker sizes using square root (between log and linear)
        if "aggregation_count" in df.columns:
            df = df.sort_values("aggregation_count", ascending=False).reset_index(drop=True)
            # Scale: base_size (15) + (sqrt(count) - 1) * factor, so count=1 stays at base_size
            marker_sizes = 15 + (np.sqrt(df["aggregation_count"]) - 1) * 3
        else:
            marker_sizes = 15

        # Helper to generate hover text for each point.
        def generate_hover_text(row):
            return "<br>".join([f"{col}: {row[col]}" for col in self.hover_columns])

        # Generate hover text for all points (unless suppressed or using molecule hover)
        suppress_hover = self.suppress_hover_display or self.has_smiles
        if suppress_hover:
            # Use "none" to hide the default hover display but still fire hoverData callbacks
            # Don't set hovertemplate when suppressing - it would override hoverinfo
            hovertext = None
            hovertemplate = None
            hoverinfo = "none"
        else:
            hovertext = df.apply(generate_hover_text, axis=1)
            hovertemplate = "%{hovertext}<extra></extra>"
            hoverinfo = None

        # Build customdata columns - include smiles and id if available for molecule hover
        custom_data_cols = list(self.custom_data) if self.custom_data else []
        if self.has_smiles:
            # Add smiles as first column, id as second (if available)
            if self.smiles_column not in custom_data_cols:
                custom_data_cols = [self.smiles_column] + custom_data_cols
            if self.id_column and self.id_column not in custom_data_cols:
                custom_data_cols.insert(1, self.id_column)

        # Determine marker settings based on the type of the color column.
        if pd.api.types.is_numeric_dtype(df[color_col]):
            marker_color = df[color_col]
            colorbar = dict(title=color_col, thickness=10)
            # Single trace for numeric data.
            data = [
                go.Scattergl(
                    x=df[x_col],
                    y=df[y_col],
                    mode="markers",
                    hoverinfo=hoverinfo,
                    hovertext=hovertext,
                    hovertemplate=hovertemplate,
                    customdata=df[custom_data_cols] if custom_data_cols else None,
                    marker=dict(
                        size=marker_sizes,
                        color=marker_color,
                        colorscale=self.colorscale,
                        colorbar=colorbar,
                        opacity=0.9,
                        line=dict(color="rgba(0,0,0,0.25)", width=1),
                    ),
                )
            ]
            showlegend = False
        else:
            # For categorical data, create one trace per category so that a legend appears.
            categories = sorted(df[color_col].astype(str).unique())

            # Discrete colorscale using Plotly Express's qualitative palette.
            discrete_colors = px.colors.qualitative.Plotly
            data = []
            for i, cat in enumerate(categories):
                sub_df = df[df[color_col] == cat]
                sub_hovertext = hovertext.loc[sub_df.index] if hovertext is not None else None
                # Get marker sizes for this subset (handles both array and scalar)
                if isinstance(marker_sizes, (pd.Series, np.ndarray)):
                    sub_marker_sizes = (
                        marker_sizes.loc[sub_df.index]
                        if isinstance(marker_sizes, pd.Series)
                        else marker_sizes[sub_df.index]
                    )
                else:
                    sub_marker_sizes = marker_sizes
                trace = go.Scattergl(
                    x=sub_df[x_col],
                    y=sub_df[y_col],
                    mode="markers",
                    name=cat,
                    hoverinfo=hoverinfo,
                    hovertext=sub_hovertext,
                    hovertemplate=hovertemplate,
                    customdata=sub_df[custom_data_cols] if custom_data_cols else None,
                    marker=dict(
                        size=sub_marker_sizes,
                        color=discrete_colors[i % len(discrete_colors)],
                        opacity=0.8,
                        line=dict(color="rgba(0,0,0,0.25)", width=1),
                    ),
                )
                data.append(trace)
            showlegend = True

        # Okay we have the data, now create the figure.
        # Note: We're going to add the prediction interval bands and regression line before
        #       the scatter plot data to ensure they appear below the scatter points.
        figure = go.Figure()
        if y_col == "prediction" or x_col == "prediction":
            figure = prediction_intervals(df, figure, x_col)

        # Add regression line if enabled.
        if regression_line:
            axis_min = min(df[x_col].min(), df[y_col].min())
            axis_max = max(df[x_col].max(), df[y_col].max())
            figure.add_shape(
                type="line",
                line=dict(width=4, color="rgba(128, 128, 128, 1.0)"),
                x0=axis_min,
                x1=axis_max,
                y0=axis_min,
                y1=axis_max,
            )

        # Now add the scatter plot data on last (on top)
        figure.add_traces(data)

        # Set up axes.
        if self.show_axes:
            xaxis = dict(
                title=dict(text=x_col, font=dict(size=16), standoff=15), tickformat=".2f", tickfont=dict(size=10)
            )
            yaxis = dict(
                title=dict(text=y_col, font=dict(size=16), standoff=25), tickformat=".2f", tickfont=dict(size=10)
            )
        else:
            xaxis = dict(visible=False)
            yaxis = dict(visible=False)

        # Update layout.
        figure.update_layout(
            margin={"t": 20, "b": 55, "r": 0, "l": 35, "pad": 0},
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
                Input(f"{self.component_id}-regression-line", "value"),
            ],
            prevent_initial_call=True,
        )
        def _update_scatter_plot(x_value, y_value, color_value, regression_line):
            """Update the Scatter Plot Graph based on the dropdown values."""

            # Check if the dataframe is not empty and the values are not None
            if not self.df.empty and x_value and y_value and color_value:
                figure = self.create_scatter_plot(self.df, x_value, y_value, color_value, regression_line)
                return figure

            raise PreventUpdate

        # Clientside callback for circle overlay - runs in browser, no server round trip
        clientside_callback(
            circle_overlay_callback(self._circle_data_uri),
            Output(f"{self.component_id}-overlay", "show"),
            Output(f"{self.component_id}-overlay", "bbox"),
            Output(f"{self.component_id}-overlay", "children"),
            Input(f"{self.component_id}-graph", "hoverData"),
        )

        @callback(
            Output(f"{self.component_id}-molecule-tooltip", "show"),
            Output(f"{self.component_id}-molecule-tooltip", "bbox"),
            Output(f"{self.component_id}-molecule-tooltip", "children"),
            Input(f"{self.component_id}-graph", "hoverData"),
        )
        def _scatter_molecule_overlay(hover_data):
            """Show molecule tooltip when smiles data is available."""
            if hover_data is None or not self.has_smiles:
                return False, no_update, no_update

            # Extract customdata (contains smiles and id)
            customdata = hover_data["points"][0].get("customdata")
            if customdata is None:
                return False, no_update, no_update

            # SMILES is the first element, ID is second (if available)
            if isinstance(customdata, (list, tuple)):
                smiles = customdata[0]
                mol_id = customdata[1] if len(customdata) > 1 and self.id_column else None
            else:
                smiles = customdata
                mol_id = None

            # Generate molecule tooltip with ID header (use cached background color)
            mol_width, mol_height = 300, 200
            children = molecule_hover_tooltip(
                smiles, mol_id=mol_id, width=mol_width, height=mol_height, background=self.hover_background
            )

            # Position molecule tooltip above and slightly right of the point
            bbox = hover_data["points"][0]["bbox"]
            center_x = (bbox["x0"] + bbox["x1"]) / 2
            center_y = (bbox["y0"] + bbox["y1"]) / 2
            x_offset = 5  # Slight offset to the right
            y_offset = mol_height + 50  # Above the point

            adjusted_bbox = {
                "x0": center_x + x_offset,
                "x1": center_x + x_offset + mol_width,
                "y0": center_y - mol_height - y_offset,
                "y1": center_y - y_offset,
            }
            return True, adjusted_bbox, children


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

    # Get a UQ regressor model
    from workbench.api import Model

    model = Model("logd-reg-xgb")
    df = model.get_inference_predictions("full_cross_fold")

    # Run the Unit Test on the Plugin
    # Test currently commented out
    """
    PluginUnitTest(
        ScatterPlot,
        input_data=df,
        theme="midnight_blue",
        x="logd",
        y="prediction",
        color="prediction_std",
        suppress_hover_display=True,
    ).run()
    """

    # Test with molecule hover (smiles column)
    from workbench.api import FeatureSet

    fs = FeatureSet("aqsol_features")
    mol_df = fs.pull_dataframe()[:1000]  # Limit to 1000 rows for testing

    # Run the Unit Test with molecule data (hover over points to see molecule structures)
    PluginUnitTest(
        ScatterPlot,
        input_data=mol_df,
        theme="midnight_blue",
        suppress_hover_display=True,
    ).run()
