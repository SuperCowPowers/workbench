"""A Violin Plot component"""

from dash import dcc
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

# Workbench Imports
from workbench.web_interface.components.component_interface import ComponentInterface


class ViolinPlots(ComponentInterface):
    """Correlation Matrix Component"""

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Violin Plot Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Violin Plot Component
        """
        return dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

    def update_properties(self, df: pd.DataFrame, figure_args: dict, max_plots: int = 40) -> go.Figure:
        """Create a set of violin plots for the numeric columns in the dataframe.
        Args:
            df (pd.DataFrame): The dataframe containing the data.
            figure_args (dict): A dictionary of arguments to pass to the plot object.
                For violin plot arguments, refer to: https://plotly.com/python/reference/violin/
            max_plots (int): The maximum number of plots to create (default: 40).
        Returns:
            go.Figure: A Figure object containing the generated plots.
        """

        # Sanity check the dataframe
        if df is None or df.empty or list(df.columns) == ["name", "status"]:
            return self.display_text("No Data Found", figure_height=200)

        numeric_columns = list(df.select_dtypes("number").columns)
        numeric_columns = [col for col in numeric_columns if df[col].nunique() > 1]  # Only columns > 1 unique value

        # HARDCODE: Not sure how to get around hard coding these columns
        not_show = [
            "id",
            "Id",
            "ID",
            "name",
            "write_time",
            "api_invocation_time",
            "is_deleted",
            "x",
            "y",
            "cluster",
            "outlier_group",
        ]
        numeric_columns = [col for col in numeric_columns if col not in not_show]
        numeric_columns = numeric_columns[:max_plots]  # Max plots

        # Compute the number of rows and columns
        num_plots = len(numeric_columns)
        num_rows, num_columns = self._compute_subplot_layout(num_plots)
        fig = make_subplots(rows=num_rows, cols=num_columns, vertical_spacing=0.07)
        for i, col in enumerate(numeric_columns):
            # Truncate labels to a maximum of 24 characters
            label = f"{col[:20]}..." if len(col) > 24 else col
            fig.add_trace(
                go.Violin(y=df[col], name=label, **figure_args),
                row=i // num_columns + 1,
                col=i % num_columns + 1,
            )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=(self._calculate_height(num_rows)),
            dragmode="select",
            newselection=dict(line=dict(color="grey", width=1, dash="dot")),
        )
        fig.update_traces(selected_marker=dict(size=10, color="white"), selector=dict(type="violin"))
        fig.update_traces(unselected_marker=dict(size=6, opacity=0.5), selector=dict(type="violin"))
        fig.update_traces(
            box_line_color="rgba(255, 255, 255, 0.75)",
            meanline_color="rgba(255, 255, 255, 0.75)",
            width=0.5,
            selector=dict(type="violin"),
        )
        return fig

    @staticmethod
    def _compute_subplot_layout(n):
        """Internal method to compute a 'nice' layout for a given number of subplots
        Args:
            n (int): The total number of subplots.
        Returns:
            tuple: A tuple (rows, cols) representing the layout.
        Logic:
            We're aiming for a rectangular grid of plots with max columns = 8
        """

        # Start with a single row
        rows = 1
        while True:
            cols = math.ceil(n / rows)
            if cols <= 8:
                return int(rows), int(cols)
            else:
                rows += 1

    @staticmethod
    def _calculate_height(num_rows: int) -> int:
        """Internal method to calculate the height of the total figure based on the number of rows
        Args:
            num_rows (int): The number of rows in the plot.
        Returns:
            int: The height of the aggregate plot.
        """
        # Set the base height
        base_height = 250
        if num_rows == 1:
            return base_height
        return base_height + num_rows * 80


if __name__ == "__main__":
    # This class takes in DataSource details and generates a Confusion Matrix
    from workbench.api.data_source import DataSource

    ds = DataSource("abalone_data")
    smart_sample_rows = ds.smart_sample()

    # Instantiate the ViolinPlots class
    v_plots = ViolinPlots()

    # Generate the figure
    fig = v_plots.update_properties(
        smart_sample_rows,
        figure_args={
            "box_visible": True,
            "meanline_visible": True,
            "showlegend": False,
            "points": "all",
            "spanmode": "hard",
        },
    )

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
