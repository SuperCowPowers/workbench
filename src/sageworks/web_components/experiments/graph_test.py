import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import networkx as nx
import numpy as np

# Generate a random graph using NetworkX
G = nx.random_geometric_graph(10, 0.5)

# Assign a random value to each node and a random weight to each edge
for node in G.nodes():
    G.nodes[node]["value"] = np.random.randint(3, 8)

for edge in G.edges():
    G.edges[edge]["weight"] = np.random.randint(1, 5)

# Compute the positions of the nodes using a force-directed layout, considering edge weights
pos = nx.spring_layout(G, weight="weight", iterations=100)

# Extract the node positions and values
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
node_values = [G.nodes[node]["value"] for node in G.nodes()]

# Create a trace for the nodes
node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    marker=dict(
        size=[value * 5 for value in node_values],
        color=node_values,
        colorscale="Viridis",
        opacity=1,
        line=dict(color="black", width=1),
        showscale=True,
        colorbar=dict(title="Node Value"),
    ),
    text=[f'Node {node}<br>Value: {G.nodes[node]["value"]}' for node in G.nodes()],
    hoverinfo="text",
)

# Create separate traces for each edge with varying width based on weight
edge_traces = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = G.edges[edge]["weight"]
    edge_trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        line=dict(width=weight, color="black"),  # Constant color, varying width
        hoverinfo="none",
        mode="lines",
    )
    edge_traces.append(edge_trace)

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(
    [
        dcc.Graph(
            id="network-graph",
            figure={
                "data": edge_traces + [node_trace],
                "layout": go.Layout(
                    title="Force-Directed Graph with Varying Edge Widths",
                    showlegend=False,
                    hovermode="closest",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                ),
            },
        )
    ]
)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
