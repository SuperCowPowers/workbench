from dash import dcc, html, callback, Input, Output
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import networkx as nx

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.theme_manager import ThemeManager
from workbench.utils.color_utils import weights_to_colors


class GraphPlot(PluginInterface):
    """A Graph Plot Plugin for NetworkX Graphs."""

    # Initialize this Plugin Component Class with required attributes
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.GRAPH

    def __init__(self):
        """Initialize the Graph (Nodes/Edges) Plugin"""
        self.component_id = None
        self.hover_columns = []
        self.graph = None
        self.graph_figure = None

        # Initialize the Theme Manager
        self.theme_manager = ThemeManager()

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
            className="workbench-container",
            children=[
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

    def update_properties(self, input_graph: nx.Graph, **kwargs) -> list:
        """Update the property values for the plugin component.

        Args:
            input_graph (NetworkX Graph): The input graph data object.
            **kwargs: Additional keyword arguments (plugins can define their own arguments).
                      Note: The current kwargs processed are:
                            - label: The default node labels
                            - color: The default color column
                            - label_columns: The columns to use for the labels
                            - color_columns: The columns to use for the color scale
                            - hover_columns: The columns to show when hovering over a node

        Returns:
            list: A list of updated properties (figure, label_list, color_list, default_label, default)color).
        """

        # Get the NetworkX graph
        self.graph = input_graph

        # We'll use the first node to look for node attributes
        first_node = self.graph.nodes[next(iter(self.graph.nodes))]

        # Check if nodes have 'x' and 'y' attributes
        if "x" not in first_node or "y" not in first_node:
            self.log.important("No 'x'/'y' attributes found, running layout...")

            # Choose layout algorithm based on graph size
            node_count = len(self.graph.nodes())
            if node_count < 500:
                self.log.important(f"Graph has {node_count} nodes, using spring layout")
                pos = nx.spring_layout(self.graph, iterations=500)
            else:
                self.log.important(f"Large graph with {node_count} nodes, using spectral layout")
                pos = nx.spectral_layout(self.graph)

            # Update x/y attributes for all nodes
            for node, position in pos.items():
                self.graph.nodes[node]["x"] = position[0]
                self.graph.nodes[node]["y"] = position[1]

        # Use 'id' as default label field if not specified
        label_field = kwargs.get("label", next(iter(first_node), "id"))

        # Add degree attribute if not already present in nodes
        nx.set_node_attributes(self.graph, dict(self.graph.degree()), "degree")

        # Extract positions, labels, degrees, and hover text in a single pass through the nodes
        x_nodes, y_nodes, labels, node_degrees, hover_text = [], [], [], [], []

        # Define hover columns if not specified
        hover_columns = kwargs.get("hover_columns", ["id", "degree"])  # Default hover columns
        if hover_columns == "all":
            hover_columns = [key for key in first_node if key != "x" and key != "y"]

        # Iterate through nodes once and extract required fields
        for node, data in self.graph.nodes(data=True):
            x_nodes.append(data["x"])
            y_nodes.append(data["y"])
            labels.append(data.get(label_field, ""))
            node_degrees.append(data.get("degree", self.graph.degree[node]))
            hover_text.append("<br>".join([f"{key}: {data.get(key, '')}" for key in hover_columns]))

        # Node label colors
        if self.theme_manager.dark_mode():
            label_color = "rgba(200, 200, 255)"
        else:
            label_color = "rgba(100, 100, 100)"

        # Create an OpenGL Scattergl plot for nodes using Plotly
        node_trace = go.Scattergl(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",  # Include text mode for labels
            text=labels,  # Set text for node labels
            textfont=dict(color=label_color, size=14),  # Set font size/color for node labels
            textposition="top center",  # Position labels above the nodes
            hovertext=hover_text,  # Set hover text for nodes
            hovertemplate="%{hovertext}<extra></extra>",  # Define hover template and remove extra info
            marker=dict(
                size=20,  # Marker size for nodes
                color=node_degrees,  # Use node degrees for marker colors
                colorbar=dict(title="Degree"),  # Include a color bar for degrees
                line=dict(color="rgba(0, 0, 0, 0.5)", width=2),  # Set border color and width for nodes
            ),
        )

        # Create Scattergl traces for edges in a single loop for efficiency
        edge_traces = []
        color_scale = self.theme_manager.colorscale()
        edge_width = 5
        edge_weights = list(nx.get_edge_attributes(self.graph, "weight").values())
        edge_colors = weights_to_colors(edge_weights, color_scale)
        for edge, color in zip(self.graph.edges(), edge_colors):
            x0, y0 = self.graph.nodes[edge[0]]["x"], self.graph.nodes[edge[0]]["y"]
            x1, y1 = self.graph.nodes[edge[1]]["x"], self.graph.nodes[edge[1]]["y"]

            # Create individual Scattergl trace for each edge with specific styling
            edge_traces += [
                go.Scattergl(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=edge_width + 2, color="rgba(0, 0, 0, 0.5)"),  # Edge drop shadow
                    showlegend=False,
                    hoverinfo="skip",  # Skip hover info for edges if not needed
                ),
                go.Scattergl(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=edge_width, color=color),
                    showlegend=False,
                    hoverinfo="skip",  # Skip hover info for edges if not needed
                ),
            ]

        # Create a Plotly figure with the combined node and edge traces
        self.graph_figure = go.Figure(data=edge_traces + [node_trace])

        # Fine-tune the plot's layout and aesthetics
        self.graph_figure.update_layout(
            margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10},  # Set margins and padding
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),  # Hide X-axis grid and tick marks
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),  # Hide Y-axis grid and tick marks
            showlegend=False,  # Remove legend
            dragmode="pan",
            modebar={"bgcolor": "rgba(0, 0, 0, 0)"},  # Transparent background for modebar
            uirevision="constant",  # Keep the layout constant for updates
        )

        # Get the first node's attributes (all fields should be defined here)
        node_attributes = {key for key in first_node.keys()}
        node_attributes.discard("x")  # Remove 'x/y' from the attributes
        node_attributes.discard("y")

        # Create dropdown options for the label list (all non-'x/y' attributes)
        label_list = [{"label": "None", "value": "None"}]  # Add "None" option as the first entry
        label_list.extend([{"label": attr, "value": attr} for attr in node_attributes])

        # Create the color list using only numeric attributes or enumerated string fields
        color_list = []
        for attr in node_attributes:
            sample_value = first_node[attr]

            # Check if the attribute is numeric
            if isinstance(sample_value, (int, float)):
                color_list.append({"label": attr, "value": attr})
            # If it's a string, add the option with a note about enumeration
            elif isinstance(sample_value, str):
                color_list.append({"label": f"{attr} (enum)", "value": attr})

        # Set default dropdown values for label and color
        default_label = label_field  # Use the specified label field as default
        default_color = "degree" if "degree" in node_attributes else next(iter(node_attributes), "id")

        # Return the updated properties for the dropdowns and the figure
        return [self.graph_figure, label_list, color_list, default_label, default_color]

    def register_internal_callbacks(self):
        """Register any internal callbacks for the plugin."""

        @callback(
            Output(f"{self.component_id}-graph", "figure", allow_duplicate=True),
            [
                Input(f"{self.component_id}-label-dropdown", "value"),
                Input(f"{self.component_id}-color-dropdown", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_graph(label, color):
            """Update the Graph (Nodes/Edges) based on the dropdown values."""
            if not label and not color:
                raise PreventUpdate

            # Use the class variable to access the graph's nodes directly
            nodes = self.graph.nodes

            # Get the node trace from the graph figure (assuming last trace is for nodes)
            node_trace = self.graph_figure["data"][-1]

            # Update node labels dynamically if a label field is selected
            if label:
                if label == "None":  # Handle "None" option
                    node_trace["text"] = [""] * len(nodes)
                else:
                    node_trace["text"] = [nodes[node].get(label, "") for node in nodes]

            # Update node colors dynamically if a color field is selected
            if color:
                # Check if the attribute exists and if it's numeric
                first_node = next(iter(nodes))
                if color in nodes[first_node]:
                    color_values = [nodes[node][color] for node in nodes]
                    if isinstance(nodes[first_node][color], (int, float)):
                        node_trace["marker"]["color"] = color_values
                    else:
                        # Enumerate strings as a fallback for categorical values
                        unique_values = {val: idx for idx, val in enumerate(set(color_values))}
                        node_trace["marker"]["color"] = [unique_values[val] for val in color_values]
                else:
                    node_trace["marker"]["color"] = [0] * len(nodes)  # Default to zero if not present

            # Return the updated figure
            return self.graph_figure


if __name__ == "__main__":
    """Run the Unit Test for the Plugin."""
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest
    from workbench.api.graph_store import GraphStore

    # Pull a test graph
    graph_store = GraphStore()
    test_graph = graph_store.get("chem_info/tox21_100")
    if test_graph is None:
        print("Test graph not found... using default")

    # Run the Unit Test on the Plugin
    PluginUnitTest(GraphPlot, theme="dark", input_data=test_graph).run()
