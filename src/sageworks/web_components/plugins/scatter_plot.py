from typing import Union
import pandas as pd
from dash import dcc
import plotly.graph_objects as go
import networkx as nx

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType

class ScatterPlot(PluginInterface):
    """A Graph Plot Plugin for NetworkX Graphs."""

    # Initialize this Plugin Component Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.FEATURE_SET

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Dash Graph Component without any data.

        Args:
            component_id (str): The ID of the web component.

        Returns:
            dcc.Graph: A Dash Graph Component.
        """
        # Fill in plugin properties and signals
        self.properties = [(f"{component_id}", "figure")]
        self.signals = [(f"{component_id}", "hoverData")]

        return dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

    def update_properties(self, input_data: Union[DataSource, FeatureSet, pd.DataFrame], **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            input_data (DataSource or FeatureSet): The input data object.
            **kwargs: Additional keyword arguments (plugins can define their own arguments).

        Returns:
            list: A list of updated property values (just [go.Figure] for now).
        """

        # Grab the dataframe from the input data object
        if isinstance(input_data, (DataSource, FeatureSet)):
            df = input_data.pull_dataframe()
        else:
            df = input_data

        # Is the label field specified in the kwargs?
        label_field = kwargs.get("labels", "id")

        # Fill in the labels
        labels = df[label_field].tolist()

        # Are the hover_text fields specified in the kwargs, if not just use the 'id' field
        hover_text_fields = kwargs.get("hover_text", ["id"])
        if hover_text_fields == "all":
            hover_text_fields = df.columns.tolist()

        # Fill in the hover text
        hover_text = [
            "<br>".join([f"{key}: {row.get(key, '')}" for key in hover_text_fields])
            for row in df.to_dict(orient="records")
        ]

        # Define a color scale for the nodes (blue, yellow, orange, red)
        color_scale = [
            [0.0, "rgb(64,64,160)"],
            [0.33, "rgb(48, 140, 140)"],
            [0.67, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"]
        ]

        # Are the field choices specified in the kwargs?
        # If not use all the numeric columns in the dataframe
        field_choices = kwargs.get("field_choices", df.select_dtypes(include="number").columns.tolist())

        # Ensure there are at least three numeric columns for x, y, and color
        if len(field_choices) < 3:
            raise ValueError("At least three numeric columns are required for x, y, and color.")

        # Create Plotly Scatter Plot
        figure = go.Figure(data=go.Scatter(
            x=df[field_choices[0]],
            y=df[field_choices[1]],
            mode="markers",
            hovertext=hover_text,  # Set hover text
            hovertemplate="%{hovertext}<extra></extra>",  # Define hover template and remove extra info
            textfont=dict(family="Arial Black", size=14),  # Set font size
            marker=dict(
                size=20,
                color=df[field_choices[2]],  # Use the third field for color
                colorscale=color_scale,
                colorbar=dict(title=field_choices[2]),
                line=dict(color="Black", width=1),
            ),
        ))

        # Just some fine-tuning of the plot
        figure.update_layout(
            margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10},
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),  # Remove X axis grid and tick marks
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),  # Remove Y axis grid and tick marks
            showlegend=False,  # Remove legend
        )

        # Apply dark theme
        # figure.update_layout(template="plotly_dark")

        # Return the figure
        return [figure]

    def register_internal_callbacks(self):
        """Register any internal callbacks for the plugin."""
        pass


if __name__ == "__main__":
    # This class takes in graph details and generates a Graph Plot (go.Figure)
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(ScatterPlot).run()
