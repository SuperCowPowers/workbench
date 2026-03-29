"""LaplacianSmoothness: Graph Laplacian-based signal smoothness analysis."""

import numpy as np
import pandas as pd
import networkx as nx
import logging

from workbench.algorithms.graph.light.proximity_graph import ProximityGraph

# Set up logging
log = logging.getLogger("workbench")


class LaplacianSmoothness:
    """Measure how smoothly a signal (e.g., target values) varies across a proximity graph.

    Uses the graph Laplacian to quantify local and global signal smoothness.
    A smooth signal means neighboring nodes have similar values; a non-smooth
    signal means neighbors disagree — indicating potential data quality issues,
    activity cliffs, or measurement noise.

    The per-node score is: score_i = Σ_j w_ij * (y_i - y_j)²
    The global score is the normalized Laplacian quadratic form: x^T L x / num_edges

    Args:
        proximity_graph (ProximityGraph): A pre-built ProximityGraph instance.
        signal (str): Node attribute name to use as the signal (e.g., target column).

    Example:
        ```python
        from workbench.algorithms.graph.light.signal_smoothness import LaplacianSmoothness
        from workbench.algorithms.graph.light.proximity_graph import ProximityGraph
        from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity

        # Build proximity graph
        prox = FeatureSpaceProximity(df, id_column="id", features=features, target="target")
        pg = ProximityGraph()
        pg.build_graph(prox, n_neighbors=5)

        # Compute signal smoothness
        ss = LaplacianSmoothness(pg, signal="target")
        print(ss.scores())
        print(f"Global smoothness: {ss.global_score():.4f}")
        non_smooth = ss.get_non_smooth(top_percent=5.0)
        ```
    """

    def __init__(self, proximity_graph: ProximityGraph, signal: str):
        self.nx_graph = proximity_graph.nx_graph
        self.signal = signal

        # Validate that the signal attribute exists on nodes
        sample_node = next(iter(self.nx_graph.nodes()))
        node_data = self.nx_graph.nodes[sample_node]
        if signal not in node_data:
            raise ValueError(f"Signal '{signal}' not found in node attributes. Available: {list(node_data.keys())}")

        # Compute per-node smoothness scores
        self._scores_df = self._compute_scores()
        log.info(f"LaplacianSmoothness computed for {len(self._scores_df)} nodes on signal '{signal}'")

    def _compute_scores(self) -> pd.DataFrame:
        """Compute per-node smoothness scores using weighted neighbor differences."""
        node_scores = []
        for node, data in self.nx_graph.nodes(data=True):
            y_i = data[self.signal]
            score = 0.0
            for neighbor in self.nx_graph.neighbors(node):
                y_j = self.nx_graph.nodes[neighbor][self.signal]
                weight = self.nx_graph.edges[node, neighbor].get("weight", 1.0)
                score += weight * (y_i - y_j) ** 2
            node_scores.append({"node": node, self.signal: y_i, "smoothness_score": score})

        df = pd.DataFrame(node_scores)
        df["smoothness_rank"] = df["smoothness_score"].rank(pct=True)
        return df.sort_values("smoothness_score", ascending=False).reset_index(drop=True)

    def scores(self) -> pd.DataFrame:
        """Return per-node smoothness scores.

        Returns:
            pd.DataFrame: DataFrame with columns [node, signal, smoothness_score, smoothness_rank]
                sorted by smoothness_score descending. Higher scores = less smooth.
        """
        return self._scores_df.copy()

    def global_score(self) -> float:
        """Compute the global signal smoothness: x^T L x normalized by number of edges.

        Returns:
            float: Global smoothness score. Higher = less smooth overall.
        """
        L = nx.laplacian_matrix(self.nx_graph, weight="weight").toarray()
        nodes = list(self.nx_graph.nodes())
        x = np.array([self.nx_graph.nodes[n][self.signal] for n in nodes], dtype=float)
        num_edges = self.nx_graph.number_of_edges()
        if num_edges == 0:
            return 0.0
        return float(x @ L @ x) / num_edges

    def get_non_smooth(self, top_percent: float = 5.0) -> pd.DataFrame:
        """Get the least smooth nodes (highest smoothness scores).

        Args:
            top_percent (float): Percentage of nodes to return (default: 5.0 = top 5%)

        Returns:
            pd.DataFrame: DataFrame of least smooth nodes, sorted by score descending.
        """
        threshold = 1.0 - (top_percent / 100.0)
        return self._scores_df[self._scores_df["smoothness_rank"] >= threshold].reset_index(drop=True)


if __name__ == "__main__":
    # Self-contained test using raw NetworkX graph (no AWS dependencies)

    class _MockProximityGraph:
        """Minimal stand-in for ProximityGraph to avoid AWS imports."""

        def __init__(self, G):
            self._nx_graph = G

        @property
        def nx_graph(self):
            return self._nx_graph

    def _make_test_graph(targets):
        """Build a simple proximity graph with given target values."""
        G = nx.Graph()
        nodes = ["a", "b", "c", "d"]
        for node, target in zip(nodes, targets):
            G.add_node(node, target=target)
        # Connect neighbors: a-b, b-c, c-d, a-c (like the Laplacian diagram)
        G.add_edge("a", "b", weight=1.0)
        G.add_edge("b", "c", weight=1.0)
        G.add_edge("c", "d", weight=1.0)
        G.add_edge("a", "c", weight=0.8)

        return _MockProximityGraph(G)

    # Smooth signal: neighbors have similar values
    smooth_pg = _make_test_graph([0.9, 1.0, 0.8, 0.9])

    # Non-smooth signal: neighbors have very different values
    noisy_pg = _make_test_graph([0.9, -0.8, 0.7, -0.5])

    for label, pg in [("SMOOTH", smooth_pg), ("NON-SMOOTH", noisy_pg)]:
        print(f"\n{'='*60}")
        print(f"  {label} SIGNAL")
        print(f"{'='*60}")

        ss = LaplacianSmoothness(pg, signal="target")
        print(f"\nGlobal smoothness score: {ss.global_score():.4f}")
        print("\nPer-node scores:")
        print(ss.scores())
        print("\nTop 50% least smooth:")
        print(ss.get_non_smooth(top_percent=50.0))

    # Integration test with FeatureSpaceProximity
    from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity

    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "f1": [1, 1.1, 1.2, 3, 3.1],
            "f2": [1, 1.1, 1.2, 3, 3.1],
            "target": [10, 10.1, 10.2, 20, 20.1],
        }
    )
    prox = FeatureSpaceProximity(df, id_column="id", features=["f1", "f2"], target="target")
    pg = ProximityGraph()
    pg.build_graph(prox, n_neighbors=3)
    ss = LaplacianSmoothness(pg, signal="target")
    print(ss.scores())
