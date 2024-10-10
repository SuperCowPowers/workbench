from typing import Union
from dash import dcc, html, callback, Input, Output
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import networkx as nx

# SageWorks Imports
from sageworks.core.artifacts.graph_core import GraphCore
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType


class GraphPlot(PluginInterface):
    """A Graph Plot Plugin for NetworkX Graphs."""

    # Initialize this Plugin Component Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.GRAPH

    def __init__(self):
        """Initialize the Graph (Nodes/Edges) Plugin"""
        self.component_id = None
        self.hover_columns = []
        self.graph_figure = None
        self.nodes = None
        self.edges = None

        # Call the parent class constructor
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Graph (Node/Edge) Component without any data.

        Args:
            component_id (str): The ID of the web component.

        Returns:
            html.Div: A Container of Components for the Graph (Node/Edge) Component.
        """
        self.component_id = component_id

        # Fill in plugin properties and signals
        self.properties = [
            (f"{component_id}-graph", "figure"),
            (f"{component_id}-label-dropdown", "options"),
            (f"{component_id}-color-dropdown", "options"),
            (f"{component_id}-label-dropdown", "value"),
            (f"{component_id}-color-dropdown", "value"),
        ]
        self.signals = [(f"{component_id}-graph", "hoverData")]

        # Create the Composite Component
        # - A Graph Node/Edge Component
        # - Dropdown for Node Labels and Colors
        return html.Div(
            [
                # Main Scatter Plot Graph
                dcc.Graph(
                    id=f"{component_id}-graph",
                    figure=self.display_text("Waiting for Data..."),
                    config={"scrollZoom": True},
                    style={"width": "100%", "height": "100%"},  # Let the graph fill its container
                ),

                # Controls: Label, Color Dropdowns, and TBD Checkbox
                html.Div(
                    [
                        html.Label("Label", style={"marginLeft": "40px", "marginRight": "5px", "fontWeight": "bold"}),
                        dcc.Dropdown(
                            id=f"{component_id}-label-dropdown",
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
                            id=f"{component_id}-tbd-checkbox",
                            options=[{"label": " TBD", "value": "show"}],
                            value=[],
                            style={"margin": "10px"},
                        ),
                    ],
                    style={"padding": "10px", "display": "flex", "gap": "10px"},
                ),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column"},  # Full viewport height
        )

    def update_properties(self, input_graph: Union[GraphCore, nx.Graph], **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            input_graph (GraphCore or NetworkX Graph): The input graph data object.
            **kwargs: Additional keyword arguments (plugins can define their own arguments).
                      Note: The current kwargs processed are:
                            - label: The default node labels
                            - color: The default color column
                            - label_columns: The columns to use for the labels
                            - color_columns: The columns to use for the color scale
                            - hover_columns: The columns to show when hovering over a node

        Returns:
            list: A list of updated property values (figure, label options, color options, default label, default color).
        """

        # Check if the input graph is a GraphCore object
        if isinstance(input_graph, GraphCore):
            # Get the NetworkX graph from the GraphCore object
            graph = input_graph.get_nx_graph()
        else:
            # The input graph is already a NetworkX graph
            graph = input_graph

        # Check to make sure the graph nodes have a 'pos' attribute
        if not all("pos" in graph.nodes[node] for node in graph.nodes):
            if len(graph.nodes) > 100:
                self.log.warning("Graph nodes do not have 'pos' attribute. Running spring layout...")
            else:
                self.log.important("Graph nodes do not have 'pos' attribute. Running spring layout...")
            pos = nx.spring_layout(graph, iterations=500)
            for node, coords in pos.items():
                graph.nodes[node]["pos"] = list(coords)

        # Extract positions for plotting
        x_nodes = [data["pos"][0] for node, data in graph.nodes(data=True)]
        y_nodes = [data["pos"][1] for node, data in graph.nodes(data=True)]

        # Check if the degree attribute exists, and if not, compute it
        first_node = next(iter(graph.nodes))
        if "degree" not in graph.nodes[first_node]:
            self.log.important("Computing node degrees...")
            degrees = dict(graph.degree())
            nx.set_node_attributes(graph, degrees, "degree")

        # Now we can extract the node degrees and define a color scale based on min/max degrees
        node_degrees = [data["degree"] for node, data in graph.nodes(data=True)]

        # Is the label field specified in the kwargs?
        label_field = kwargs.get("labels", "id")

        # Fill in the labels
        labels = [data.get(label_field, "") for node, data in graph.nodes(data=True)]

        # Are the hover_text fields specified in the kwargs?
        hover_columns = kwargs.get("hover_columns", ["id", "degree"])
        if hover_columns == "all":
            # All fields except for 'pos'
            hover_columns = [key for key in graph.nodes[first_node] if key != "pos"]

        # Fill in the hover text
        hover_text = [
            "<br>".join([f"{key}: {data.get(key, '')}" for key in hover_columns])
            for node, data in graph.nodes(data=True)
        ]

        # Define a custom color scale (blue -> yellow -> orange -> red)
        color_scale = [
            [0.0, "rgb(64, 64, 160)"],
            [0.33, "rgb(48, 140, 140)"],
            [0.67, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"],
        ]

        # Create an OpenGL Scatter Plot for our nodes
        node_trace = go.Scattergl(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",  # Include text mode for labels
            text=labels,  # Set text for labels
            textposition="top center",  # Position labels
            hovertext=hover_text,  # Set hover text
            hovertemplate="%{hovertext}<extra></extra>",  # Define hover template and remove extra info
            textfont=dict(family="Arial Black", size=14),  # Set font size
            marker=dict(
                size=20,
                color=node_degrees,
                colorscale=color_scale,
                colorbar=dict(title="Degree"),
                line=dict(color="Black", width=1),
            ),
        )

        # Create OpenGL Scattergl traces for edges
        edge_traces = []
        for edge in graph.edges():
            x0, y0 = graph.nodes[edge[0]]["pos"]
            x1, y1 = graph.nodes[edge[1]]["pos"]

            # Check for edge weight and set defaults
            weight = graph.edges[edge].get("weight", 0.5)

            # Scale the width and alpha of the edge based on the weight
            width = min(5.0, weight * 4.9 + 0.1)  # Scale to [0.1, 5]
            alpha = min(1.0, weight * 0.9 + 0.1)  # Scale to [0.1, 1.0]

            # Create individual Scattergl trace for each edge with specific styling
            edge_traces.append(
                go.Scattergl(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=width, color=f"rgba(150, 150, 150, {alpha})"),
                    showlegend=False,
                    hoverinfo='skip',  # Skip hover info for edges if not needed
                )
            )

        # Create figure with combined node and edge traces
        self.graph_figure = go.Figure(data=edge_traces + [node_trace])

        # Just some fine-tuning of the plot
        plotly_theme = "plotly_dark" if self.dark_theme else "plotly"
        self.graph_figure.update_layout(
            template=plotly_theme,
            margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10},
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),  # Remove X axis grid and tick marks
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),  # Remove Y axis grid and tick marks
            showlegend=False,  # Remove legend
            dragmode="pan",
        )

        # In update_properties or another method
        self.nodes = {node: data for node, data in graph.nodes(data=True)}
        self.edges = list(graph.edges(data=True))

        # Extract all unique node attribute fields
        node_attributes = set()
        for _, data in graph.nodes(data=True):
            node_attributes.update(data.keys())

        # Create dropdown options using the unique node attributes
        label_list = [{"label": attr, "value": attr} for attr in node_attributes]
        color_list = label_list.copy()  # Use the same attributes for colors initially

        # Set default values
        default_label = label_field  # Keep the current `label_field` as default
        default_color = "degree" if "degree" in node_attributes else next(iter(node_attributes), "id")

        # Return the updated properties for the dropdowns and the figure
        return [self.graph_figure, label_list, color_list, default_label, default_color]

    def register_internal_callbacks(self):
        """Register any internal callbacks for the plugin."""

        def update_graph(label, color):
            """Update the Graph (Nodes/Edges) based on the dropdown values."""
            if not label and not color:
                raise PreventUpdate

            # Use class variables to access nodes directly
            nodes = self.nodes  # Use pre-stored node attributes

            # Get the node trace from the graph figure
            node_trace = self.graph_figure['data'][-1]  # Assumes node trace is the last trace

            # Update node labels dynamically if a label field is selected
            if label:
                node_trace['text'] = [nodes[node].get(label, "") for node in nodes]

            # Update node colors dynamically if a color field is selected
            if color:
                node_trace['marker']['color'] = [nodes[node].get(color, 0) for node in nodes]

            # Return the updated figure
            return self.graph_figure


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(GraphPlot).run()
