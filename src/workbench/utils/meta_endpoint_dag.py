"""``MetaEndpointDAG`` — a directed acyclic graph of endpoints and
aggregation nodes describing an inference-time data flow.

A DAG has two kinds of nodes:

- **Endpoint nodes** — references to deployed Workbench ``Endpoint``
  instances by name. The DAG defers actual ``Endpoint`` instantiation
  until execution / column-contract resolution.

- **Aggregation nodes** — instances of :class:`AggregationNode` subclasses
  that combine outputs from upstream nodes.

DAG construction is explicit::

    dag = MetaEndpointDAG()
    dag.add_endpoint("smiles-to-2d-v1")
    dag.add_endpoint("smiles-to-3d-fast-v1")
    dag.add_aggregation(Concat(name="combine"))
    dag.add_edge("smiles-to-2d-v1", "combine")
    dag.add_edge("smiles-to-3d-fast-v1", "combine")
    dag.set_input_node("smiles-to-2d-v1", "smiles-to-3d-fast-v1")
    dag.set_output_node("combine")
    dag.validate()

Row-alignment across parallel branches: the walker injects a synthetic
:data:`DAG_ROW_ID` column at the start of every ``run()`` and strips it
before returning. Aggregation nodes use it as the join key, so callers
do not need to supply (or care about) any id column on their input data.

Validation runs at construction time so misconfigured DAGs fail loud
before any inference round-trips.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

import pandas as pd

from workbench.utils.aggregation_nodes import DAG_ROW_ID, AggregationNode

EndpointInvoker = Callable[[str, pd.DataFrame], pd.DataFrame]


class MetaEndpointDAG:
    """A typed DAG of endpoints + aggregation nodes.

    The DAG joins parallel branches using an internal synthetic row id
    (:data:`DAG_ROW_ID`) injected by :meth:`run` — callers don't need to
    supply any id column on their input.
    """

    def __init__(self):
        self._endpoints: Dict[str, str] = {}  # node_name → endpoint_name
        self._endpoint_async_flags: Dict[str, bool] = {}  # populated by populate_async_flags()
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

    @property
    def endpoints(self) -> Dict[str, str]:
        """Mapping of node_name → endpoint_name (read-only view)."""
        return self._endpoints

    @property
    def aggregations(self) -> Dict[str, AggregationNode]:
        """Mapping of node_name → :class:`AggregationNode` (read-only view)."""
        return self._aggregations

    @property
    def input_nodes(self) -> List[str]:
        """Node names that receive the DAG's input DataFrame directly (read-only)."""
        return self._input_nodes

    @property
    def output_node(self) -> Optional[str]:
        """Node name whose output is the DAG's output."""
        return self._output_node

    @property
    def endpoint_async_flags(self) -> Dict[str, bool]:
        """Mapping of endpoint_name → is_async (populated by :meth:`populate_async_flags`)."""
        return self._endpoint_async_flags

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

    def input_columns(self) -> List[str]:
        """Union of input columns required by every node that receives the
        caller's input directly.

        Used by :class:`MetaEndpoint` as a fallback when deriving the
        feature list during lineage anchoring.
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

        for name in self._aggregations:
            if not self._parents_of(name):
                raise ValueError(f"Aggregation node '{name}' has no upstream parents")

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

    def run(
        self,
        input_df: pd.DataFrame,
        endpoint_invoker: Optional[EndpointInvoker] = None,
    ) -> pd.DataFrame:
        """Execute the DAG against ``input_df`` and return the output node's DataFrame.

        The walker injects a synthetic :data:`DAG_ROW_ID` column at entry
        (used internally to align rows across parallel branches) and
        strips it before returning. Callers don't need to supply any id
        column.

        Walks nodes in topological order. Endpoint nodes call
        :meth:`Endpoint.inference` on either the caller's ``input_df`` (input
        nodes) or their upstream parent's cached output. Aggregation nodes
        receive the cached outputs of all their parents and apply their
        combination logic.

        Failure policy is fail-fast: any exception in any node propagates
        out and the DAG run aborts.

        Args:
            input_df: DataFrame supplied by the caller. Must contain the
                columns required by every input-node endpoint. No id
                column is required.
            endpoint_invoker: Optional callable ``(endpoint_name, df) -> df``
                used to invoke endpoint nodes. Defaults to using the full
                Workbench ``Endpoint`` API class — appropriate for client-side
                use. Pass a ``fast_inference``-backed invoker when running
                inside a deployed SageMaker container where the full
                Workbench config isn't available.

        Returns:
            The DataFrame at the DAG's output node, with the synthetic
            :data:`DAG_ROW_ID` column removed.
        """
        if self._output_node is None:
            raise ValueError("DAG has no output node — call set_output_node() first")
        if DAG_ROW_ID in input_df.columns:
            raise ValueError(
                f"input_df already contains the reserved column '{DAG_ROW_ID}'. " f"Remove it before calling run()."
            )

        # Inject the synthetic row id. Endpoints will pass this through as an
        # unknown input column; aggregation nodes use it as their join key.
        input_df = input_df.copy()
        input_df[DAG_ROW_ID] = range(len(input_df))

        outputs: Dict[str, pd.DataFrame] = {}
        for node in self.topological_order():
            if node in self._endpoints:
                outputs[node] = self._run_endpoint(node, input_df, outputs, endpoint_invoker)
            else:
                outputs[node] = self._run_aggregation(node, outputs)

        result = outputs[self._output_node]
        if DAG_ROW_ID in result.columns:
            result = result.drop(columns=[DAG_ROW_ID])
        return result

    def _run_endpoint(
        self,
        node: str,
        input_df: pd.DataFrame,
        outputs: Dict[str, pd.DataFrame],
        endpoint_invoker: Optional[EndpointInvoker],
    ) -> pd.DataFrame:
        """Execute a single endpoint node.

        Source DataFrame is the caller's input for input nodes, or the
        single upstream parent's output otherwise. The full DataFrame is
        passed to ``endpoint.inference()`` — metadata columns
        (project_id, owner, etc.) flow through alongside the endpoint's
        added columns, matching standard Workbench inference behavior.

        The walker-injected :data:`DAG_ROW_ID` column must survive the
        endpoint round-trip so downstream aggregation nodes can join on
        it. If an endpoint silently strips unknown input columns, this
        will fail loudly — better than misaligned rows.
        """
        endpoint_name = self._endpoints[node]
        parents = self._parents_of(node)
        source_df = input_df if not parents else outputs[parents[0]]

        if endpoint_invoker is not None:
            result = endpoint_invoker(endpoint_name, source_df)
        else:
            from workbench.api import Endpoint

            result = Endpoint(endpoint_name).inference(source_df)

        if DAG_ROW_ID not in result.columns:
            raise RuntimeError(
                f"Endpoint '{endpoint_name}' dropped the walker-injected '{DAG_ROW_ID}' "
                f"column from its output. The DAG can't align rows across branches "
                f"without it. Endpoints must pass unknown input columns through to "
                f"their output."
            )
        return result

    def _run_aggregation(self, node: str, outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute a single aggregation node."""
        agg = self._aggregations[node]
        upstream = [outputs[p] for p in self._parents_of(node)]
        return agg.apply(upstream)

    # ------------------------------------------------------------------
    # Serialization (model artifact + workbench_meta storage)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the DAG topology to a JSON-friendly dict.

        Aggregation nodes are serialized by class name + constructor kwargs;
        deserialization (:meth:`from_dict`) requires the same class to be
        importable.

        Per-endpoint ``is_async`` flags are included only if
        :meth:`populate_async_flags` has been called. The deployed
        inference container relies on these flags to dispatch invocations
        to ``fast_inference`` or ``async_inference``.
        """
        return {
            "endpoints": dict(self._endpoints),
            "endpoint_async": dict(self._endpoint_async_flags),
            "aggregations": [_serialize_aggregation(a) for a in self._aggregations.values()],
            "edges": [list(e) for e in self._edges],
            "input_nodes": list(self._input_nodes),
            "output_node": self._output_node,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def populate_async_flags(self) -> None:
        """Look up each endpoint's async flag via ``workbench_meta`` and store it.

        Flags are keyed by endpoint name (not node name) so the deployed
        invoker can dispatch directly on the value passed by the walker.

        Called by :meth:`MetaEndpoint.create` before serializing the DAG
        for deployment. Hits AWS once per unique endpoint name, so isolated
        as an explicit step rather than running implicitly in :meth:`to_dict`.
        """
        from workbench.api import Endpoint

        for endpoint_name in set(self._endpoints.values()):
            meta = Endpoint(endpoint_name).workbench_meta() or {}
            self._endpoint_async_flags[endpoint_name] = bool(meta.get("async_endpoint"))

    def has_async_endpoint(self) -> bool:
        """Return True if any endpoint in the DAG is deployed as async.

        Used by :meth:`MetaEndpoint.create` to decide whether the meta
        endpoint itself must be deployed as async. Lazily calls
        :meth:`populate_async_flags` if not yet populated.
        """
        if not self._endpoint_async_flags and self._endpoints:
            self.populate_async_flags()
        return any(self._endpoint_async_flags.values())

    def terminal_target(self) -> Optional[str]:
        """Target column the DAG ultimately predicts, or ``None`` for feature pipelines.

        - Output is an endpoint → return its model's target.
        - Output is an aggregation → walk back to the closest endpoint(s),
          collect their targets. Returns the unique target if all agree;
          ``None`` if zero predictors are upstream or their targets disagree.

        Used by :meth:`MetaEndpoint._derive_lineage` to anchor the meta's
        ``target_column`` on what the DAG actually predicts rather than
        inheriting from the (possibly target-less) input endpoint.
        """
        from workbench.api import Endpoint, Model

        def _target_of(ep_name: str) -> Optional[str]:
            ep = Endpoint(ep_name)
            if not ep.exists():
                return None
            return Model(ep.get_input()).target()

        if self._output_node in self._endpoints:
            return _target_of(self._endpoints[self._output_node])

        # Aggregation output — BFS back until we hit endpoints, collecting targets.
        targets: set = set()
        seen: set = set()
        queue: List[str] = list(self._parents_of(self._output_node))
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            if node in self._endpoints:
                t = _target_of(self._endpoints[node])
                if t:
                    targets.add(t)
            else:
                queue.extend(self._parents_of(node))

        if len(targets) == 1:
            return targets.pop()
        return None

    @classmethod
    def from_dict(cls, data: dict) -> "MetaEndpointDAG":
        dag = cls()
        for node_name, endpoint_name in data.get("endpoints", {}).items():
            dag.add_endpoint(endpoint_name, node_name=node_name)
        dag._endpoint_async_flags = dict(data.get("endpoint_async", {}))
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
    state = {"class": type(node).__name__, "name": node.name}
    for attr in ("weights", "model_weights", "corr_scale", "optimal_alpha"):
        if hasattr(node, attr):
            val = getattr(node, attr)
            try:
                state[attr] = val.tolist()
            except AttributeError:
                state[attr] = val
    return state


def _deserialize_aggregation(data: dict) -> AggregationNode:
    # Dual-resolve: workbench-package path (normal) and bare path (container).
    try:
        from workbench.utils import aggregation_nodes as agg_module
    except ImportError:  # pragma: no cover — SageMaker container only
        import aggregation_nodes as agg_module

    cls_name = data["class"]
    cls = getattr(agg_module, cls_name)
    kwargs = {k: v for k, v in data.items() if k != "class"}
    return cls(**kwargs)
