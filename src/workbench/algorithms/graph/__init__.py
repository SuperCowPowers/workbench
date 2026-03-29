"""Welcome to the Graph Algorithm Classes

These classes provide functionality for Graphs (NetworkX).

- ProximityGraph: Build a NetworkX graph from a Proximity instance
- LaplacianSmoothness: Graph Laplacian-based signal smoothness analysis
"""

from .light.proximity_graph import ProximityGraph
from .light.laplacian_smoothness import LaplacianSmoothness

__all__ = ["ProximityGraph", "LaplacianSmoothness"]
