from dash import dcc
import plotly.graph_objects as go
import networkx as nx

# SageWorks Imports
from sageworks.core.artifacts.graph_core import GraphCore
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType


class GraphPlot(PluginInterface):
    """A Graph Plot Plugin for NetworkX Graphs."""

    # Initialize this Plugin Component Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.GRAPH

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Dash Graph Component without any data.

        Args:
            component_id (str): The ID of the web component.

        Returns:
            dcc.Graph: A Dash Graph Component.
        """
        # Fill in plugin properties
        self.properties = [(f"{component_id}", "figure")]

        return dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

    def update_properties(self, input_graph: GraphCore , **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            input_graph (GraphCore): The input graph data object.
            **kwargs: Additional keyword arguments (plugins can define their own arguments).

        Returns:
            list: A list of updated property values (just [go.Figure] for now).
        """
        # Get the NetworkX graph from our GraphCore object
        graph = input_graph.get_nx_graph()

        # Check to make sure the graph nodes have a 'pos' attribute
        if not all('pos' in graph.nodes[node] for node in graph.nodes):
            self.log.warning("Graph nodes do not have 'pos' attribute. Running spring layout...")
            pos = nx.spring_layout(graph, dim=2)
            input_graph.store_node_positions(pos)
            graph = input_graph.get_nx_graph()

        # Extract positions for plotting
        x_nodes = [data['pos'][0] for node, data in graph.nodes(data=True)]
        y_nodes = [data['pos'][1] for node, data in graph.nodes(data=True)]

        # Check if the degree attribute exists, and if not, compute it
        if 'degree' not in graph.nodes[0]:
            self.log.important("Computing node degrees...")
            degrees = dict(graph.degree())
            nx.set_node_attributes(graph, degrees, 'degree')

        # Now we can extract the node degrees and define a color scale based on min/max degrees
        node_degrees = [data['degree'] for node, data in graph.nodes(data=True)]

        # Define a color scale for the nodes based on a normalized node degree
        color_scale = [
            [0.0, "rgb(64,64,160)"],
            [0.33, "rgb(48, 140, 140)"],
            [0.67, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"],
        ]

        # Create Plotly trace for nodes
        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers',
            marker=dict(
                size=16,
                color=node_degrees,
                colorscale=color_scale,
                colorbar=dict(title="Degree"),
                line=dict(color="Black", width=1)
            )
        )

        # Create Plotly traces for edges
        edge_traces = []
        for edge in graph.edges():
            x0, y0 = graph.nodes[edge[0]]['pos']
            x1, y1 = graph.nodes[edge[1]]['pos']

            # Check to see if edge has a weight attribute
            if 'weight' in graph.edges[edge]:
                weight = graph.edges[edge]['weight']
            else:
                weight = 0.5

            # Scale the width and alpha of the edge
            width = min(5.0, weight * 4.9 + 0.1)  # Scale to [0.1, 5]
            alpha = min(1.0, weight * 0.9 + 0.1)  # Scale to [0.1, 1.0]

            # Create the edge trace and append to the list
            edge_traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                          line=dict(width=width, color=f"rgba(150, 150, 150, {alpha})")))

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        # Just some fine-tuning of the plot
        fig.update_layout(
            margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10},
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),  # Remove X axis grid and tick marks
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),  # Remove Y axis grid and tick marks
            showlegend=False  # Remove legend
        )

        # Apply dark theme
        fig.update_layout(template="plotly_dark")

        # Return the figure
        return [fig]

    def register_internal_callbacks(self):
        """Register any internal callbacks for the plugin."""
        pass


if __name__ == "__main__":
    # This class takes in graph details and generates a Graph Plot (go.Figure)
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(GraphPlot).run()
