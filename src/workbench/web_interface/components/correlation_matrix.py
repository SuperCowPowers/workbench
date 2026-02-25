"""A Correlation Matrix component"""

from dash import dcc
import plotly.graph_objects as go
import pandas as pd

from workbench.web_interface.components.component_interface import ComponentInterface
from workbench.utils.theme_manager import ThemeManager
from workbench.utils.color_utils import remove_middle_colors


class CorrelationMatrix(ComponentInterface):
    """Correlation Matrix Component (specialized Plotly Heatmap)

    See: https://plotly.com/python/heatmaps/
    """

    def __init__(self):
        """Initialize the Correlation Matrix Class"""
        self.theme_manager = ThemeManager()
        self.colorscale = remove_middle_colors(self.theme_manager.colorscale("diverging"))
        super().__init__()

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Correlation Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Correlation Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

    def update_properties(self, data_source_details: dict) -> go.Figure:
        """Create a Correlation Matrix Figure for the numeric columns in the dataframe.

        Args:
            data_source_details (dict): A dictionary containing DataSource details.

        Returns:
            plotly.graph_objs.Figure: A Figure object containing the correlation matrix.
        """
        if data_source_details is None:
            return self.display_text("No Details Data Found", figure_height=200)
        if "column_stats" not in data_source_details:
            return self.display_text("No column_stats Found", figure_height=200)

        # For small numbers of numeric columns, show all correlations
        column_stats = data_source_details["column_stats"]
        num_numeric = sum(1 for info in column_stats.values() if "correlations" in info)
        threshold = 0.0 if num_numeric < 6 else 0.4

        df = self._corr_df_from_data_details(data_source_details, threshold=threshold)
        if df.empty:
            return self.display_text("No Correlations Found", figure_height=200)

        # Heatmap has inverse y-axis ordering, so flip the dataframe
        df = df.iloc[::-1]

        # Store column indexes in labels (workaround for getting click-point index)
        x_labels = [f"{c}:{i}" for i, c in enumerate(df.columns)]
        y_labels = [f"{c}:{i}" for i, c in enumerate(df.index)]

        # For small matrices (low threshold), use data-driven color range
        z_range = max(df.abs().max().max(), 0.1) if num_numeric < 6 else 1

        height = max(350, len(df.index) * 50)
        fig = go.Figure(
            data=go.Heatmap(
                z=df,
                x=x_labels,
                y=y_labels,
                xgap=2,
                ygap=2,
                name="",
                colorscale=self.colorscale,
                zmin=-z_range,
                zmax=z_range,
            )
        )
        fig.update_layout(margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 0}, height=height)

        # Truncate labels to 24 characters
        x_axes_labels = [f"{c[:20]}..." if len(c) > 24 else c for c in df.columns]
        y_axes_labels = [f"{c[:20]}..." if len(c) > 24 else c for c in df.index]
        fig.update_xaxes(tickvals=x_labels, ticktext=x_axes_labels, tickangle=30, showgrid=False)
        fig.update_yaxes(tickvals=y_labels, ticktext=y_axes_labels, showgrid=False)

        # Annotate cells above threshold
        for i, row in enumerate(df.index):
            for j, col in enumerate(df.columns):
                value = df.loc[row, col]
                if abs(value) > threshold:
                    fig.add_annotation(x=j, y=i, text=f"{value:.2f}", showarrow=False)

        return fig

    @staticmethod
    def _corr_df_from_data_details(data_details: dict, threshold: float = 0.4) -> pd.DataFrame:
        """Internal: Create a correlation DataFrame from DataSource details.
        Args:
            data_details (dict): A dictionary containing DataSource details.
            threshold (float): Any correlations below this value will be excluded.
        Returns:
            pd.DataFrame: A Pandas DataFrame containing the correlation matrix
        """
        if not data_details:
            return pd.DataFrame()

        column_stats = data_details["column_stats"]
        corr_dict = {key: info["correlations"] for key, info in column_stats.items() if "correlations" in info}
        corr_df = pd.DataFrame(corr_dict).fillna(0)

        # Filter out correlations below threshold, iteratively raising it if matrix > 12x12
        corr_df = corr_df.loc[:, corr_df.abs().max() > threshold]
        corr_df = corr_df[corr_df.abs().max(axis=1) > threshold]
        while corr_df.shape[0] > 12:
            threshold += 0.05
            corr_df = corr_df.loc[:, corr_df.abs().max() > threshold]
            corr_df = corr_df[corr_df.abs().max(axis=1) > threshold]

        corr_df = corr_df.sort_index()
        corr_df = corr_df[corr_df.index]
        return corr_df


if __name__ == "__main__":
    from workbench.api.data_source import DataSource

    tm = ThemeManager()
    tm.set_theme("dark")

    ds = DataSource("test_data")
    corr_plot = CorrelationMatrix()
    fig = corr_plot.update_properties(ds.details())
    fig.show()
