"""A confusion matrix plugin component"""
from dash import Dash, dcc, no_update
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import random


# SageWorks Imports
from sageworks.web_components.component_interface import ComponentInterface


class ConfusionMatrix(ComponentInterface):
    """Confusion Matrix Component"""

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion MatrixComponent without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    def generate_component_figure(self, df: pd.DataFrame = None) -> go.Figure:
        """Create a Correlation Matrix Figure for the numeric columns in the dataframe.
        Args:
            df (pd.DataFrame): The dataframe containing the data for the correlation matrix.
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the correlation matrix.
        """
        if df is None:
            # Make some fake data
            data = {
                'low': [random.uniform(0.5, 0.9), 0.2, 0.3],
                'med': [0.1, random.uniform(0.5, 0.9), 0.2],
                'high': [0.1, 0.1, random.uniform(0.5, 0.9)]
            }
            index_labels = ['low', 'med', 'high']
            df = pd.DataFrame(data, index=index_labels)

        # A nice color scale for the correlation matrix
        color_scale = [
            [0, "rgb(64,64,200)"],
            [0.35, "rgb(48, 180, 180)"],
            [0.65, "rgb(180, 180, 48)"],
            [1.0, "rgb(200, 64, 64)"],
        ]

        # Okay so the heatmap has inverse y-axis ordering, so we need to flip the dataframe
        # df = df.iloc[::-1]

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
                name="",
                colorscale=color_scale,
                zmin=0,
                zmax=1,
            )
        )
        fig.update_layout(margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 0}, height=height)

        # Now remap the x and y axis labels (so they don't show the index)
        fig.update_xaxes(tickvals=x_labels, ticktext=df.columns, tickangle=30, tickfont_size=14)
        fig.update_yaxes(tickvals=y_labels, ticktext=df.index, tickfont_size=14)

        # Now we're going to customize the annotations
        for i, row in enumerate(df.index):
            for j, col in enumerate(df.columns):
                value = df.loc[row, col]
                fig.add_annotation(x=j, y=i, text=f"{value:.2f}", showarrow=False, font_size=14)

        return fig

    # Register the callbacks for the component
    def register_callbacks(self, app: Dash):
        self.update_confusion_matrix(app)

    # Updates the confusion matrix when a model row is selected
    def update_confusion_matrix(self, app: Dash):
        @app.callback(
            Output(self.component_id(), "figure"),
            Input("models_table", "derived_viewport_selected_row_ids"),
            prevent_initial_call=True,
        )
        def update_callback(selected_rows):
            print(f"Selected Rows: {selected_rows}")
            if not selected_rows or selected_rows[0] is None:
                return no_update

            return self.generate_component_figure()
