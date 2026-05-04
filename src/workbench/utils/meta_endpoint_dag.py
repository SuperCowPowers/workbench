"""``MetaEndpointDAG`` — a directed acyclic graph of endpoints and
aggregation nodes describing an inference-time data flow.

A DAG has two kinds of nodes:

- **Endpoint nodes** — references to deployed Workbench ``Endpoint``
  instances by name. The DAG defers actual ``Endpoint`` instantiation
  until execution / column-contract resolution.

- **Aggregation nodes** — instances of :class:`AggregationNode` subclasses
  that combine outputs from upstream nodes.

DAG construction is explicit::

    dag = MetaEndpointDAG(id_column="id")
    dag.add_endpoint("smiles-to-2d-v1")
    dag.add_endpoint("smiles-to-3d-fast-v1")
    dag.add_aggregation(Concat(name="combine", id_column="id"))
    dag.add_edge("smiles-to-2d-v1", "combine")
    dag.add_edge("smiles-to-3d-fast-v1", "combine")
    dag.set_input_node("smiles-to-2d-v1", "smiles-to-3d-fast-v1")
    dag.set_output_node("combine")
    dag.validate()

Validation runs at construction time so misconfigured DAGs fail loud
before any inference round-trips.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import pandas as pd

from workbench.utils.aggregation_nodes import AggregationNode


class MetaEndpointDAG:
    """A typed DAG of endpoints + aggregation nodes.

    Args:
        id_column: Name of the column used to join across nodes (default
            ``"id"``). Must be present on the caller's input DataFrame and
            on every endpoint output.
    """

    def __init__(self, id_column: str = "id"):
        self.id_column = id_column
        self._endpoints: Dict[str, str] = {}  # node_name → endpoint_name
        self._aggregations: Dict[str, AggregationNode] = {}
        self._edges: List[tuple[str, str]] = []  # (from_node, to_node)
        self._input_nodes: List[str] = []
        self._output_node: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_endpoint(self, endpoint_name: str, node_name: Optional[str] = None) -> str:
        """Add an endpoint reference to the DAG.

        Args:
            endpoint_name: Name of a deployed Workbench endpoint.
            node_name: Optional unique node name (defaults to ``endpoint_name``).

        Returns:
            The node name (so callers can chain).
        """
        node = node_name or endpoint_name
        if node in self._endpoints or node in self._aggregations:
            raise ValueError(f"Node '{node}' already exists in this DAG")
        self._endpoints[node] = endpoint_name
        return node

    def add_aggregation(self, node: AggregationNode) -> str:
        """Add an :class:`AggregationNode` to the DAG.

        The node's ``name`` must be unique across the DAG.
        """
        if node.name in self._endpoints or node.name in self._aggregations:
            raise ValueError(f"Node '{node.name}' already exists in this DAG")
        self._aggregations[node.name] = node
        return node.name

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Declare data flow from ``from_node`` to ``to_node``.

        Endpoint nodes accept at most one inbound edge (their input DataFrame
        comes from a single upstream producer). Aggregation nodes can have
        any number of inbound edges.
        """
        if from_node not in self._all_nodes():
            raise ValueError(f"Edge from unknown node '{from_node}'")
        if to_node not in self._all_nodes():
            raise ValueError(f"Edge to unknown node '{to_node}'")
        if to_node in self._endpoints and self._parents_of(to_node):
            raise ValueError(
                f"Endpoint node '{to_node}' already has an upstream parent "
                f"('{self._parents_of(to_node)[0]}'); endpoints take input "
                f"from at most one source."
            )
        self._edges.append((from_node, to_node))

    def set_input_node(self, *nodes: str) -> None:
        """Declare which nodes receive the DAG's input DataFrame directly."""
        for n in nodes:
            if n not in self._endpoints:
                raise ValueError(f"Input nodes must be endpoint nodes; '{n}' is not")
        self._input_nodes = list(nodes)

    def set_output_node(self, node: str) -> None:
        """Declare the terminal node whose output is the DAG's output."""
        if node not in self._all_nodes():
            raise ValueError(f"Unknown output node '{node}'")
        self._output_node = node

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def _all_nodes(self) -> List[str]:
        return list(self._endpoints.keys()) + list(self._aggregations.keys())

    def _parents_of(self, node: str) -> List[str]:
        return [src for src, dst in self._edges if dst == node]

    def topological_order(self) -> List[str]:
        """Return nodes in topological order (parents before children).

        Raises:
            ValueError: If the DAG contains a cycle.
        """
        in_degree = {n: 0 for n in self._all_nodes()}
        for _, dst in self._edges:
            in_degree[dst] += 1

        ready = [n for n, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for src, dst in self._edges:
                if src == node:
                    in_degree[dst] -= 1
                    if in_degree[dst] == 0:
                        ready.append(dst)

        if len(order) != len(in_degree):
            raise ValueError("DAG contains a cycle")
        return order

    # ------------------------------------------------------------------
    # Column contract
    # ------------------------------------------------------------------

    def output_columns(self) -> List[str]:
        """Static analysis: the columns the DAG emits at its output node.

        Resolves each node's contribution by walking the topology. For
        endpoint nodes this calls ``Endpoint(name).output_columns()``;
        for aggregation nodes this asks the node what columns it emits
        given its upstream column lists.
        """
        from workbench.api import Endpoint

        if self._output_node is None:
            raise ValueError("DAG has no output node — call set_output_node() first")

        per_node: Dict[str, List[str]] = {}
        for node in self.topological_order():
            if node in self._endpoints:
                per_node[node] = Endpoint(self._endpoints[node]).output_columns()
            else:
                parents = self._parents_of(node)
                upstream_outputs = [per_node[p] for p in parents]
                per_node[node] = self._aggregations[node].output_columns(upstream_outputs)
        return per_node[self._output_node]

    def input_columns(self) -> List[str]:
        """Static analysis: the union of input columns required by every
        node that receives the caller's input directly.
        """
        from workbench.api import Endpoint

        if not self._input_nodes:
            raise ValueError("DAG has no input nodes — call set_input_node() first")

        seen = set()
        cols: List[str] = []
        for node in self._input_nodes:
            for c in Endpoint(self._endpoints[node]).input_columns():
                if c not in seen:
                    seen.add(c)
                    cols.append(c)
        return cols

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> "MetaEndpointDAG":
        """Validate the DAG. Returns self for chaining; raises on failure.

        Checks:
          - At least one input node and exactly one output node declared
          - No cycles
          - Aggregation nodes have at least one parent
          - Endpoint nodes are either input nodes (zero parents) or have
            exactly one upstream parent — never both
          - The output node is reachable from the input nodes
          - Every aggregation node's ``id_column`` matches the DAG's
        """
        if not self._input_nodes:
            raise ValueError("DAG has no input nodes")
        if self._output_node is None:
            raise ValueError("DAG has no output node")

        order = self.topological_order()  # raises on cycle

        for ep_node in self._endpoints:
            parents = self._parents_of(ep_node)
            is_input = ep_node in self._input_nodes
            if is_input and parents:
                raise ValueError(
                    f"Endpoint node '{ep_node}' is declared as an input node but has "
                    f"upstream parents {parents}; pick one or the other."
                )
            if not is_input and not parents:
                raise ValueError(
                    f"Endpoint node '{ep_node}' has no upstream parent and is not "
                    f"declared as an input node — it has no source for its input DataFrame."
                )

        for name, agg in self._aggregations.items():
            parents = self._parents_of(name)
            if not parents:
                raise ValueError(f"Aggregation node '{name}' has no upstream parents")
            if agg.id_column != self.id_column:
                raise ValueError(
                    f"Aggregation node '{name}' has id_column='{agg.id_column}' "
                    f"but DAG id_column='{self.id_column}'"
                )

        reachable = set(self._input_nodes)
        for node in order:
            if node in reachable:
                for src, dst in self._edges:
                    if src == node:
                        reachable.add(dst)
        if self._output_node not in reachable:
            raise ValueError(f"Output node '{self._output_node}' is not reachable from input nodes {self._input_nodes}")

        return self

    # ------------------------------------------------------------------
    # Execution (client-side walker)
    # ------------------------------------------------------------------

    def run(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Execute the DAG against ``input_df`` and return the output node's DataFrame.

        Walks nodes in topological order. Endpoint nodes call
        :meth:`Endpoint.inference` on either the caller's ``input_df`` (input
        nodes) or their upstream parent's cached output. Aggregation nodes
        receive the cached outputs of all their parents and apply their
        combination logic.

        Failure policy is fail-fast: any exception in any node propagates
        out and the DAG run aborts.

        Args:
            input_df: DataFrame supplied by the caller. Must contain
                ``self.id_column`` and the columns required by every
                input-node endpoint.

        Returns:
            The DataFrame at the DAG's output node.
        """
        if self._output_node is None:
            raise ValueError("DAG has no output node — call set_output_node() first")
        if self.id_column not in input_df.columns:
            raise ValueError(f"input_df is missing id_column '{self.id_column}'")

        outputs: Dict[str, pd.DataFrame] = {}
        for node in self.topological_order():
            if node in self._endpoints:
                outputs[node] = self._run_endpoint(node, input_df, outputs)
            else:
                outputs[node] = self._run_aggregation(node, outputs)

        return outputs[self._output_node]

    def _run_endpoint(self, node: str, input_df: pd.DataFrame, outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute a single endpoint node.

        Source DataFrame is the caller's input for input nodes, or the
        single upstream parent's output otherwise. The full DataFrame is
        passed to ``endpoint.inference()`` — metadata columns
        (project_id, owner, etc.) flow through alongside the endpoint's
        added columns, matching standard Workbench inference behavior.
        """
        from workbench.api import Endpoint

        endpoint = Endpoint(self._endpoints[node])
        parents = self._parents_of(node)
        source_df = input_df if not parents else outputs[parents[0]]
        return endpoint.inference(source_df)

    def _run_aggregation(self, node: str, outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute a single aggregation node."""
        agg = self._aggregations[node]
        upstream = [outputs[p] for p in self._parents_of(node)]
        return agg.apply(upstream)

    # ------------------------------------------------------------------
    # Serialization (Phase 3 will use this for the model artifact)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the DAG topology to a JSON-friendly dict.

        Aggregation nodes are serialized by class name + constructor kwargs;
        deserialization (:meth:`from_dict`) requires the same class to be
        importable.
        """
        return {
            "id_column": self.id_column,
            "endpoints": dict(self._endpoints),
            "aggregations": [_serialize_aggregation(a) for a in self._aggregations.values()],
            "edges": [list(e) for e in self._edges],
            "input_nodes": list(self._input_nodes),
            "output_node": self._output_node,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "MetaEndpointDAG":
        dag = cls(id_column=data.get("id_column", "id"))
        for node_name, endpoint_name in data.get("endpoints", {}).items():
            dag.add_endpoint(endpoint_name, node_name=node_name)
        for agg_data in data.get("aggregations", []):
            dag.add_aggregation(_deserialize_aggregation(agg_data))
        for src, dst in data.get("edges", []):
            dag.add_edge(src, dst)
        if data.get("input_nodes"):
            dag.set_input_node(*data["input_nodes"])
        if data.get("output_node"):
            dag.set_output_node(data["output_node"])
        return dag

    @classmethod
    def from_json(cls, payload: str) -> "MetaEndpointDAG":
        return cls.from_dict(json.loads(payload))


def _serialize_aggregation(node: AggregationNode) -> dict:
    """Capture the node's class name + reconstructible state.

    Subclasses store their constructor kwargs as plain attributes; we
    pluck them off the instance for round-tripping. Numpy arrays
    (``model_weights``, ``corr_scale``) are converted back to lists.
    """
    state = {"class": type(node).__name__, "name": node.name, "id_column": node.id_column}
    for attr in ("weights", "model_weights", "corr_scale", "optimal_alpha"):
        if hasattr(node, attr):
            val = getattr(node, attr)
            try:
                state[attr] = val.tolist()
            except AttributeError:
                state[attr] = val
    return state


def _deserialize_aggregation(data: dict) -> AggregationNode:
    from workbench.utils import aggregation_nodes as agg_module

    cls_name = data["class"]
    cls = getattr(agg_module, cls_name)
    kwargs = {k: v for k, v in data.items() if k != "class"}
    return cls(**kwargs)
