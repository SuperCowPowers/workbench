"""A Correlation Matrix component"""

from dash import dcc
import plotly.graph_objects as go
import pandas as pd

# SageWorks Imports
from sageworks.web_interface.components.component_interface import ComponentInterface
from sageworks.utils.theme_manager import ThemeManager


# This class is basically a specialized version of a Plotly Heatmap
# For heatmaps see (https://plotly.com/python/heatmaps/)
class CorrelationMatrix(ComponentInterface):
    """Correlation Matrix Component"""

    def __init__(self):
        """Initialize the Regression Plot Class"""

        # Initialize the Theme Manager
        self.theme_manager = ThemeManager()

        # Call the parent class constructor
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

        # Sanity check the data
        if data_source_details is None:
            return self.display_text("No Details Data Found", figure_height=200)
        if "column_stats" not in data_source_details:
            return self.display_text("No column_stats Found", figure_height=200)

        # Convert the data details into a correlation dataframe
        df = self._corr_df_from_data_details(data_source_details)

        # If the dataframe is empty then return a message
        if df.empty:
            return self.display_text("No Correlations Found", figure_height=200)

        # Okay so the heatmap has inverse y-axis ordering, so we need to flip the dataframe
        df = df.iloc[::-1]

        # Okay so there are numerous issues with getting back the index of the clicked on point
        # so we're going to store the indexes of the columns (this is SO stupid)
        x_labels = [f"{c}:{i}" for i, c in enumerate(df.columns)]
        y_labels = [f"{c}:{i}" for i, c in enumerate(df.index)]

        # Create the heatmap plot with custom settings
        height = max(400, len(df.index) * 50)
        fig = go.Figure(
            data=go.Heatmap(
                z=df,
                x=x_labels,
                y=y_labels,
                xgap=2,  # Add space between cells
                ygap=2,
                name="",
                colorscale=self.theme_manager.colorscale("diverging"),
                zmin=-1,
                zmax=1,
            )
        )
        fig.update_layout(margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 0}, height=height)

        # Now remap the x and y axis labels (so they don't show the index)
        fig.update_xaxes(tickvals=x_labels, ticktext=df.columns, tickangle=30, showgrid=False)
        fig.update_yaxes(tickvals=y_labels, ticktext=df.index, showgrid=False)

        # Now we're going to customize the annotations and filter out low values
        label_threshold = 0.3
        for i, row in enumerate(df.index):
            for j, col in enumerate(df.columns):
                value = df.loc[row, col]
                if abs(value) > label_threshold:
                    fig.add_annotation(x=j, y=i, text=f"{value:.2f}", showarrow=False)

        return fig

    @staticmethod
    def _corr_df_from_data_details(data_details: dict, threshold: float = 0.3) -> pd.DataFrame:
        """Internal: Create a Pandas DataFrame in the form given by df.corr() from DataSource details.
        Args:
            data_details (dict): A dictionary containing DataSource details.
            threshold (float): Any correlations below this value will be excluded.
        Returns:
            pd.DataFrame: A Pandas DataFrame containing the correlation matrix
        """

        # Sanity check
        if not data_details:
            return pd.DataFrame()

        # Process the data so that we can make a Dataframe of the correlation data
        column_stats = data_details["column_stats"]
        corr_dict = {key: info["correlations"] for key, info in column_stats.items() if "correlations" in info}
        corr_df = pd.DataFrame(corr_dict)

        # The diagonal will be NaN, so fill it with 0
        corr_df.fillna(0, inplace=True)

        # Now filter out any correlations below the threshold
        corr_df = corr_df.loc[:, (corr_df.abs().max() > threshold)]
        corr_df = corr_df[(corr_df.abs().max(axis=1) > threshold)]

        # If the correlation matrix is bigger than 10x10 then we need to filter it down
        while corr_df.shape[0] > 10 and threshold <= 0.6:
            # Now filter out any correlations below the threshold
            corr_df = corr_df.loc[:, (corr_df.abs().max() > threshold)]
            corr_df = corr_df[(corr_df.abs().max(axis=1) > threshold)]
            threshold += 0.1

        # Return the correlation dataframe in the form of df.corr()
        corr_df.sort_index(inplace=True)
        corr_df = corr_df[corr_df.index]
        return corr_df


if __name__ == "__main__":
    # This class takes in data details and generates a Correlation Matrix
    from sageworks.api.data_source import DataSource

    tm = ThemeManager()
    tm.set_theme("dark")

    ds = DataSource("test_data")
    ds_details = ds.details()

    # Instantiate the CorrelationMatrix class
    corr_plot = CorrelationMatrix()

    # Generate the figure
    fig = corr_plot.update_properties(ds_details)

    # Apply dark theme
    fig.update_layout()

    # Show the figure
    fig.show()
