"""AQSol proximity graph: walk from the least to the most soluble compound."""

from workbench.api import Model
from workbench.algorithms.graph import shortest_path, path_subgraph
from workbench.utils.graph_utils import graph_layout
from workbench.web_interface.components.plugins.graph_plot import GraphPlot

if __name__ == "__main__":

    # The model's feature-space proximity -> NetworkX graph
    model = Model("aqsol-regression")
    prox = model.prox("features")
    target = model.target()
    graph = prox.graph(n_neighbors=5)
    print(f"Proximity graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # The extremes of the solubility range
    df = prox.df
    least = df.loc[df[target].idxmin(), prox.id_column]
    most = df.loc[df[target].idxmax(), prox.id_column]
    print(f"Least soluble: {least} ({df[target].min():.2f})")
    print(f"Most soluble:  {most} ({df[target].max():.2f})")

    # Walk between them, hopping through the most similar compounds
    path = shortest_path(graph, least, most)
    if not path:
        raise SystemExit("No path between the extremes — try a larger n_neighbors.")
    print(f"\nPath ({len(path)} compounds):")
    for node in path:
        print(f"  {node}: {graph.nodes[node][target]:.2f}")

    # Plot the path plus a hop of surrounding compounds, colored by solubility
    context = path_subgraph(graph, path, radius=1)
    print(f"\nPath subgraph: {context.number_of_nodes()} nodes, {context.number_of_edges()} edges")
    fig = GraphPlot().update_properties(graph_layout(context), label="id", color=target, hover_columns="all")[0]
    fig.show()
