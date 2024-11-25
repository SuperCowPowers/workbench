"""A confusion matrix plugin component"""

from dash import dcc
import plotly.graph_objects as go


# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.cached.cached_model import CachedModel


class ConfusionMatrix(PluginInterface):
    """Confusion Matrix Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.MODEL

    def __init__(self):
        """Initialize the ModelDetails plugin class"""
        self.component_id = None

        # Call the parent class constructor
        super().__init__()

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        self.component_id = component_id
        self.container = dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

        # Fill in plugin properties
        self.properties = [(self.component_id, "figure")]
        # self.signals = [(f"{self.component_id}", "value")]

        # Return the container
        return self.container

    def update_properties(self, model: CachedModel, **kwargs) -> list:
        """Create a Confusion Matrix Figure for the numeric columns in the dataframe.
        Args:
            model (CachedModel): Sageworks instantiated CachedModel object
            **kwargs:
                - inference_run (str): Inference capture UUID
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the confusion matrix.
        """

        # Get the inference run from the kwargs
        inference_run = kwargs.get("inference_run", "auto_inference")

        # Grab the confusion matrix from the model details
        df = model.confusion_matrix(inference_run)
        if df is None:
            return self.display_text("No Data")

        # A nice color scale for the confusion matrix
        color_scale = [
            [0, "rgb(64,64,160)"],
            [0.35, "rgb(48, 140, 140)"],
            [0.65, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"],
        ]

        # Okay so the heatmap has inverse y-axis ordering so we need to flip the dataframe
        df = df.iloc[::-1]

        # Okay so there are numerous issues with getting back the index of the clicked on point
        # so we're going to store the indexes of the columns (this is SO stupid)
        x_labels = [f"{c}:{i}" for i, c in enumerate(df.columns)]
        y_labels = [f"{c}:{i}" for i, c in enumerate(df.index)]

        # Create the heatmap plot with custom settings
        fig = go.Figure(
            data=go.Heatmap(
                z=df,
                x=x_labels,
                y=y_labels,
                xgap=2,
                ygap=2,
                name="",
                colorscale=color_scale,
                zmin=0,
            )
        )
        fig.update_layout(
            margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10},
            height=400,
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )

        # Now remap the x and y-axis labels (so they don't show the index)
        fig.update_xaxes(
            tickvals=x_labels,
            ticktext=df.columns,
            tickangle=30,
            tickfont_size=14,
            automargin=True,
            title_standoff=20,
            title_font={"size": 18, "color": "#9999cc"},
        )
        fig.update_yaxes(
            tickvals=y_labels,
            ticktext=df.index,
            tickfont_size=14,
            automargin=True,
            title_standoff=20,
            title_font={"size": 18, "color": "#9999cc"},
        )

        # Now we're going to customize the annotations
        for i, row in enumerate(df.index):
            for j, col in enumerate(df.columns):
                value = df.loc[row, col]

                # For floats, we want to show 2 decimal places
                text_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                fig.add_annotation(x=j, y=i, text=text_value, showarrow=False, font_size=16, font_color="#dddddd")

        return [fig]  # update_properties returns a list of updated property values


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    model = CachedModel("wine-classification")
    PluginUnitTest(ConfusionMatrix, input_data=model).run()
