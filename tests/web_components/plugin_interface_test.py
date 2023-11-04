"""Tests if plugin interface works with correct and incorrect inputs and returns"""
from dash import dcc
import plotly.graph_objects as go
import pandas as pd
from typing import Union, get_args


# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginType, PluginInputType
from sageworks.web_components.component_interface import ComponentInterface

class CorrectInputs(PluginInterface):
    """Subclass of PluginInterface with correct inputs and returns."""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    def generate_component_figure(self, figure_input: PluginInputType) -> go.Figure:
        """Create a Confusion Matrix Figure for the numeric columns in the dataframe.
        Args:
            figure_input (PluginInputType.MODEL_DETAILS): Model class details attribute
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the confusion matrix.
        """

        # A nice color scale for the confusion matrix
        color_scale = [
            [0, "rgb(64,64,160)"],
            [0.35, "rgb(48, 140, 140)"],
            [0.65, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"],
        ]

        # Grab the confusion matrix from the model details
        confusion_matrix = figure_input.get("confusion_matrix", dict())
        df = pd.DataFrame(confusion_matrix)

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
        fig.update_layout(margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10}, height=height)

        # Now remap the x and y axis labels (so they don't show the index)
        fig.update_xaxes(tickvals=x_labels, ticktext=df.columns, tickangle=30, tickfont_size=14)
        fig.update_yaxes(tickvals=y_labels, ticktext=df.index, tickfont_size=14)

        # Now we're going to customize the annotations
        for i, row in enumerate(df.index):
            for j, col in enumerate(df.columns):
                value = df.loc[row, col]
                fig.add_annotation(x=j, y=i, text=f"{value:.2f}", showarrow=False, font_size=14)

        return fig
    
class IncorrectNamedInputs(PluginInterface):
    """Subclass of PluginInterface with incorrectly named inputs."""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

    # Component is an incorrectly named keyword argument
    def create_component(self, component: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.
        Args:
            component (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component, figure=self.waiting_figure())

    def generate_component_figure(self, figure_input: PluginInputType) -> go.Figure:
        """Create a Confusion Matrix Figure for the numeric columns in the dataframe.
        Args:
            figure_input (PluginInputType): Model class details attribute
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the confusion matrix.
        """

        # A nice color scale for the confusion matrix
        color_scale = [
            [0, "rgb(64,64,160)"],
            [0.35, "rgb(48, 140, 140)"],
            [0.65, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"],
        ]

        # Grab the confusion matrix from the model details
        confusion_matrix = figure_input.get("confusion_matrix", dict())
        df = pd.DataFrame(confusion_matrix)

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
        fig.update_layout(margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10}, height=height)

        # Now remap the x and y axis labels (so they don't show the index)
        fig.update_xaxes(tickvals=x_labels, ticktext=df.columns, tickangle=30, tickfont_size=14)
        fig.update_yaxes(tickvals=y_labels, ticktext=df.index, tickfont_size=14)

        # Now we're going to customize the annotations
        for i, row in enumerate(df.index):
            for j, col in enumerate(df.columns):
                value = df.loc[row, col]
                fig.add_annotation(x=j, y=i, text=f"{value:.2f}", showarrow=False, font_size=14)

        return fig
    
class IncorrectTypedInputs(PluginInterface):
    """Subclass of PluginInterface with incorrectly typed inputs."""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Confusion Matrix Component without any data.
        Args:
            component (str): The ID of the web component
        Returns:
            dcc.Graph: The Confusion Matrix Component
        """
        return dcc.Graph(id=component_id, figure=self.waiting_figure())

    # figure_input: dict is incorrectly typed (PluginInputType)
    def generate_component_figure(self, figure_input: dict) -> go.Figure:
        """Create a Confusion Matrix Figure for the numeric columns in the dataframe.
        Args:
            figure_input (dict): Model class details attribute
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the confusion matrix.
        """

        # A nice color scale for the confusion matrix
        color_scale = [
            [0, "rgb(64,64,160)"],
            [0.35, "rgb(48, 140, 140)"],
            [0.65, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"],
        ]

        # Grab the confusion matrix from the model details
        confusion_matrix = figure_input.get("confusion_matrix", dict())
        df = pd.DataFrame(confusion_matrix)

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
        fig.update_layout(margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10}, height=height)

        # Now remap the x and y axis labels (so they don't show the index)
        fig.update_xaxes(tickvals=x_labels, ticktext=df.columns, tickangle=30, tickfont_size=14)
        fig.update_yaxes(tickvals=y_labels, ticktext=df.index, tickfont_size=14)

        # Now we're going to customize the annotations
        for i, row in enumerate(df.index):
            for j, col in enumerate(df.columns):
                value = df.loc[row, col]
                fig.add_annotation(x=j, y=i, text=f"{value:.2f}", showarrow=False, font_size=14)

        return fig

if __name__ == "__main__":
    
    # Incorrectly named input
    subclass_cond = issubclass(IncorrectNamedInputs, PluginInterface)
    print(f"Incorrect names is a subclass of PluginInterface?: {subclass_cond}")
    print('\n')

    # # Asserts
    # assert subclass_cond

    # Incorrectly typed input
    subclass_cond = issubclass(IncorrectTypedInputs, PluginInterface)
    print(f"Incorrect types is a subclass of PluginInterface?: {subclass_cond}")
    print('\n')

    # # Asserts
    # assert subclass_cond

    # Correct
    subclass_cond = issubclass(CorrectInputs, PluginInterface)
    print(f"Correct is a subclass of PluginInterface?: {subclass_cond}")
    print('\n')

    # # Asserts
    # assert subclass_cond
