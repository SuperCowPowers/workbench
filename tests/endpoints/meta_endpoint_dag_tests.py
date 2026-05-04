"""Unit tests for ``MetaEndpointDAG`` — focused on construction, validation,
and the client-side walker.

These tests do not require AWS credentials. The walker is exercised by
building DAGs whose endpoint nodes are stubbed via a small in-process
fake (see ``_FakeEndpoint`` below) so we test the topology + aggregation
logic without making real inference calls.

For an end-to-end test that uses real Workbench endpoints, see
``tests/feature_endpoints/test_meta_endpoint_pipeline.py`` (Phase 3).
"""

from __future__ import annotations

import pandas as pd
import pytest

from workbench.utils.aggregation_nodes import (
    Concat,
    Mean,
    Vote,
    WeightedMean,
)
from workbench.utils.meta_endpoint_dag import MetaEndpointDAG

# ---------------------------------------------------------------------------
# Fake endpoint for stubbing inference
# ---------------------------------------------------------------------------


class _FakeEndpoint:
    """Stand-in for ``workbench.api.Endpoint`` in unit tests.

    Records the input rows it received and returns a deterministic output
    DataFrame defined at construction time.
    """

    def __init__(self, name: str, output: pd.DataFrame, input_cols: list[str]):
        self.name = name
        self._output = output
        self._input_cols = input_cols
        self.received: pd.DataFrame | None = None

    def input_columns(self):
        return list(self._input_cols)

    def output_columns(self):
        return [c for c in self._output.columns if c != "id"]

    def inference(self, df: pd.DataFrame) -> pd.DataFrame:
        self.received = df.copy()
        # Mirror real Workbench endpoint behavior: input columns pass through
        # alongside the new output columns. Metadata flows naturally.
        new_cols = [c for c in self._output.columns if c != "id" and c not in df.columns]
        return df.merge(self._output[["id"] + new_cols], on="id", how="inner")


def _patch_endpoints(monkeypatch, endpoints: dict[str, _FakeEndpoint]):
    """Replace ``workbench.api.Endpoint`` resolution in ``meta_endpoint_dag``
    with the supplied fakes."""
    import workbench.api as workbench_api

    def fake_endpoint(name: str):
        if name not in endpoints:
            raise KeyError(f"No fake endpoint registered for '{name}'")
        return endpoints[name]

    monkeypatch.setattr(workbench_api, "Endpoint", fake_endpoint)


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------


def test_validate_rejects_dag_without_input_node():
    dag = MetaEndpointDAG()
    dag.add_endpoint("ep-a")
    dag.set_output_node("ep-a")
    with pytest.raises(ValueError, match="no input nodes"):
        dag.validate()


def test_validate_rejects_dag_without_output_node():
    dag = MetaEndpointDAG()
    dag.add_endpoint("ep-a")
    dag.set_input_node("ep-a")
    with pytest.raises(ValueError, match="no output node"):
        dag.validate()


def test_validate_rejects_cycle():
    dag = MetaEndpointDAG()
    dag.add_endpoint("ep-a")
    dag.add_aggregation(Concat(name="c1"))
    dag.add_aggregation(Concat(name="c2"))
    dag.add_edge("ep-a", "c1")
    dag.add_edge("c1", "c2")
    dag.add_edge("c2", "c1")
    dag.set_input_node("ep-a")
    dag.set_output_node("c2")
    with pytest.raises(ValueError, match="cycle"):
        dag.validate()


def test_validate_rejects_endpoint_with_no_source():
    dag = MetaEndpointDAG()
    dag.add_endpoint("ep-a")
    dag.add_endpoint("ep-orphan")
    dag.set_input_node("ep-a")
    dag.set_output_node("ep-a")
    with pytest.raises(ValueError, match="no upstream parent and is not"):
        dag.validate()


def test_validate_rejects_id_column_mismatch():
    dag = MetaEndpointDAG(id_column="molecule_id")
    dag.add_endpoint("ep-a")
    dag.add_aggregation(Concat(name="c1", id_column="id"))  # wrong id_column
    dag.add_edge("ep-a", "c1")
    dag.set_input_node("ep-a")
    dag.set_output_node("c1")
    with pytest.raises(ValueError, match="id_column"):
        dag.validate()


def test_endpoint_with_two_parents_rejected():
    dag = MetaEndpointDAG()
    dag.add_endpoint("a")
    dag.add_endpoint("b")
    dag.add_endpoint("predictor")
    dag.add_edge("a", "predictor")
    with pytest.raises(ValueError, match="already has an upstream parent"):
        dag.add_edge("b", "predictor")


# ---------------------------------------------------------------------------
# Topological order
# ---------------------------------------------------------------------------


def test_topological_order_diamond():
    dag = MetaEndpointDAG()
    dag.add_endpoint("a")
    dag.add_endpoint("b")
    dag.add_aggregation(Concat(name="merge"))
    dag.add_edge("a", "merge")
    dag.add_edge("b", "merge")
    dag.set_input_node("a", "b")
    dag.set_output_node("merge")
    dag.validate()
    order = dag.topological_order()
    assert order.index("a") < order.index("merge")
    assert order.index("b") < order.index("merge")


# ---------------------------------------------------------------------------
# Walker — feature pipeline pattern
# ---------------------------------------------------------------------------


def test_run_concat_pipeline(monkeypatch):
    """[2D] + [3D] → Concat is the canonical feature-pipeline shape."""
    fake_2d = _FakeEndpoint(
        "ep-2d",
        pd.DataFrame({"id": [1, 2, 3], "f2d_a": [0.1, 0.2, 0.3], "f2d_b": [10, 20, 30]}),
        input_cols=["smiles"],
    )
    fake_3d = _FakeEndpoint(
        "ep-3d",
        pd.DataFrame({"id": [1, 2, 3], "f3d_x": [1.0, 2.0, 3.0]}),
        input_cols=["smiles"],
    )
    _patch_endpoints(monkeypatch, {"ep-2d": fake_2d, "ep-3d": fake_3d})

    dag = MetaEndpointDAG()
    dag.add_endpoint("ep-2d")
    dag.add_endpoint("ep-3d")
    dag.add_aggregation(Concat(name="combine"))
    dag.add_edge("ep-2d", "combine")
    dag.add_edge("ep-3d", "combine")
    dag.set_input_node("ep-2d", "ep-3d")
    dag.set_output_node("combine")
    dag.validate()

    input_df = pd.DataFrame({"id": [1, 2, 3], "smiles": ["CCO", "c1ccccc1", "CCN"]})
    out = dag.run(input_df)

    # smiles flows through (real endpoints pass input columns through),
    # plus the new feature columns from each leaf.
    assert set(out.columns) == {"id", "smiles", "f2d_a", "f2d_b", "f3d_x"}
    assert len(out) == 3
    assert list(out.sort_values("id")["f2d_a"]) == [0.1, 0.2, 0.3]
    assert list(out.sort_values("id")["f3d_x"]) == [1.0, 2.0, 3.0]


def test_run_passes_metadata_through(monkeypatch):
    """Metadata columns (project_id, owner, etc.) flow through the endpoint
    alongside the endpoint's added feature columns, matching real Workbench
    inference behavior."""
    fake = _FakeEndpoint(
        "ep-2d",
        pd.DataFrame({"id": [1, 2], "feature_x": [0.5, 0.7]}),
        input_cols=["smiles"],
    )
    _patch_endpoints(monkeypatch, {"ep-2d": fake})

    dag = MetaEndpointDAG()
    dag.add_endpoint("ep-2d")
    dag.set_input_node("ep-2d")
    dag.set_output_node("ep-2d")
    dag.validate()

    input_df = pd.DataFrame(
        {
            "id": [1, 2],
            "smiles": ["CCO", "CCN"],
            "project_id": ["P1", "P2"],
            "owner": ["alice", "bob"],
        }
    )
    out = dag.run(input_df)

    # Endpoint receives the full DataFrame — no pre-slicing.
    assert set(fake.received.columns) == {"id", "smiles", "project_id", "owner"}

    # Metadata flows through to the output, alongside the new feature column.
    assert set(out.columns) == {"id", "smiles", "project_id", "owner", "feature_x"}
    assert list(out.sort_values("id")["project_id"]) == ["P1", "P2"]
    assert list(out.sort_values("id")["owner"]) == ["alice", "bob"]


# ---------------------------------------------------------------------------
# Walker — ensemble pattern
# ---------------------------------------------------------------------------


def test_run_mean_ensemble(monkeypatch):
    """Three predictors → Mean: prediction is row-wise average; std reflects spread."""
    pred_a = _FakeEndpoint(
        "model-a",
        pd.DataFrame({"id": [1, 2], "prediction": [1.0, 2.0], "confidence": [0.8, 0.6]}),
        input_cols=["x"],
    )
    pred_b = _FakeEndpoint(
        "model-b",
        pd.DataFrame({"id": [1, 2], "prediction": [3.0, 4.0], "confidence": [0.9, 0.5]}),
        input_cols=["x"],
    )
    pred_c = _FakeEndpoint(
        "model-c",
        pd.DataFrame({"id": [1, 2], "prediction": [5.0, 6.0], "confidence": [0.7, 0.4]}),
        input_cols=["x"],
    )
    _patch_endpoints(monkeypatch, {"model-a": pred_a, "model-b": pred_b, "model-c": pred_c})

    dag = MetaEndpointDAG()
    dag.add_endpoint("model-a")
    dag.add_endpoint("model-b")
    dag.add_endpoint("model-c")
    dag.add_aggregation(Mean(name="ensemble"))
    dag.add_edge("model-a", "ensemble")
    dag.add_edge("model-b", "ensemble")
    dag.add_edge("model-c", "ensemble")
    dag.set_input_node("model-a", "model-b", "model-c")
    dag.set_output_node("ensemble")
    dag.validate()

    out = dag.run(pd.DataFrame({"id": [1, 2], "x": [10.0, 20.0]})).sort_values("id").reset_index(drop=True)

    # Row 0: predictions [1, 3, 5] → mean 3, std ≈ 1.633
    # Row 1: predictions [2, 4, 6] → mean 4, std ≈ 1.633
    assert list(out["prediction"]) == [3.0, 4.0]
    assert all(abs(v - 1.6329931618554518) < 1e-9 for v in out["prediction_std"])


def test_run_weighted_mean(monkeypatch):
    pred_a = _FakeEndpoint("a", pd.DataFrame({"id": [1], "prediction": [10.0], "confidence": [1.0]}), input_cols=["x"])
    pred_b = _FakeEndpoint("b", pd.DataFrame({"id": [1], "prediction": [20.0], "confidence": [1.0]}), input_cols=["x"])
    _patch_endpoints(monkeypatch, {"a": pred_a, "b": pred_b})

    dag = MetaEndpointDAG()
    dag.add_endpoint("a")
    dag.add_endpoint("b")
    dag.add_aggregation(WeightedMean(name="wm", weights=[3.0, 1.0]))
    dag.add_edge("a", "wm")
    dag.add_edge("b", "wm")
    dag.set_input_node("a", "b")
    dag.set_output_node("wm")
    dag.validate()

    out = dag.run(pd.DataFrame({"id": [1], "x": [0.0]}))
    # weights normalize to (0.75, 0.25) → 0.75*10 + 0.25*20 = 12.5
    assert out["prediction"].iloc[0] == 12.5


def test_run_vote_classifier(monkeypatch):
    pred_a = _FakeEndpoint("a", pd.DataFrame({"id": [1, 2], "prediction": ["pos", "neg"]}), input_cols=["x"])
    pred_b = _FakeEndpoint("b", pd.DataFrame({"id": [1, 2], "prediction": ["pos", "pos"]}), input_cols=["x"])
    pred_c = _FakeEndpoint("c", pd.DataFrame({"id": [1, 2], "prediction": ["neg", "neg"]}), input_cols=["x"])
    _patch_endpoints(monkeypatch, {"a": pred_a, "b": pred_b, "c": pred_c})

    dag = MetaEndpointDAG()
    dag.add_endpoint("a")
    dag.add_endpoint("b")
    dag.add_endpoint("c")
    dag.add_aggregation(Vote(name="vote"))
    dag.add_edge("a", "vote")
    dag.add_edge("b", "vote")
    dag.add_edge("c", "vote")
    dag.set_input_node("a", "b", "c")
    dag.set_output_node("vote")
    dag.validate()

    out = dag.run(pd.DataFrame({"id": [1, 2], "x": [0.0, 0.0]})).sort_values("id").reset_index(drop=True)

    # Row 0: pos, pos, neg → pos (2/3)
    # Row 1: neg, pos, neg → neg (2/3)
    assert list(out["prediction"]) == ["pos", "neg"]
    assert all(abs(v - 2 / 3) < 1e-9 for v in out["confidence"])


# ---------------------------------------------------------------------------
# Walker — feature pipeline → predictor pattern
# ---------------------------------------------------------------------------


def test_run_features_then_predictor(monkeypatch):
    """[2D] + [3D] → Concat → predictor demonstrates feature pipeline feeding a model.

    Metadata columns (``project_id``) added by the caller flow all the way
    through the DAG and appear in the final prediction output.
    """
    fake_2d = _FakeEndpoint(
        "ep-2d",
        pd.DataFrame({"id": [1, 2], "f2d": [0.1, 0.2]}),
        input_cols=["smiles"],
    )
    fake_3d = _FakeEndpoint(
        "ep-3d",
        pd.DataFrame({"id": [1, 2], "f3d": [1.0, 2.0]}),
        input_cols=["smiles"],
    )
    fake_predictor = _FakeEndpoint(
        "predictor",
        pd.DataFrame({"id": [1, 2], "prediction": [42.0, 99.0], "confidence": [0.95, 0.88]}),
        input_cols=["f2d", "f3d"],
    )
    _patch_endpoints(
        monkeypatch,
        {"ep-2d": fake_2d, "ep-3d": fake_3d, "predictor": fake_predictor},
    )

    dag = MetaEndpointDAG()
    dag.add_endpoint("ep-2d")
    dag.add_endpoint("ep-3d")
    dag.add_aggregation(Concat(name="features"))
    dag.add_endpoint("predictor")
    dag.add_edge("ep-2d", "features")
    dag.add_edge("ep-3d", "features")
    dag.add_edge("features", "predictor")
    dag.set_input_node("ep-2d", "ep-3d")
    dag.set_output_node("predictor")
    dag.validate()

    input_df = pd.DataFrame({"id": [1, 2], "smiles": ["CCO", "CCN"], "project_id": ["P1", "P2"]})
    out = dag.run(input_df)

    assert list(out.sort_values("id")["prediction"]) == [42.0, 99.0]

    # Metadata propagates through the entire pipeline.
    assert "project_id" in out.columns
    assert list(out.sort_values("id")["project_id"]) == ["P1", "P2"]

    # Predictor sees everything that flowed downstream — including metadata.
    assert {"id", "smiles", "project_id", "f2d", "f3d"}.issubset(fake_predictor.received.columns)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_json_roundtrip_preserves_topology():
    dag = MetaEndpointDAG()
    dag.add_endpoint("ep-2d")
    dag.add_endpoint("ep-3d")
    dag.add_aggregation(WeightedMean(name="ensemble", weights=[2.0, 1.0]))
    dag.add_edge("ep-2d", "ensemble")
    dag.add_edge("ep-3d", "ensemble")
    dag.set_input_node("ep-2d", "ep-3d")
    dag.set_output_node("ensemble")
    dag.validate()

    payload = dag.to_json()
    rebuilt = MetaEndpointDAG.from_json(payload)
    rebuilt.validate()

    assert rebuilt.topological_order() == dag.topological_order()
    assert rebuilt._aggregations["ensemble"].weights.tolist() == dag._aggregations["ensemble"].weights.tolist()


def test_run_with_custom_endpoint_invoker():
    """A caller-supplied invoker bypasses the workbench.api.Endpoint path
    entirely — the mechanism the deployed SageMaker container will use to
    swap in ``fast_inference``."""
    captured: dict[str, pd.DataFrame] = {}

    def fake_invoker(endpoint_name: str, df: pd.DataFrame) -> pd.DataFrame:
        captured[endpoint_name] = df.copy()
        # Mirror real-endpoint pass-through: input columns + added column.
        added = pd.DataFrame({"id": df["id"], f"out_{endpoint_name}": [1.0] * len(df)})
        new_cols = [c for c in added.columns if c != "id" and c not in df.columns]
        return df.merge(added[["id"] + new_cols], on="id", how="inner")

    dag = MetaEndpointDAG()
    dag.add_endpoint("ep-a")
    dag.add_endpoint("ep-b")
    dag.add_aggregation(Concat(name="merge"))
    dag.add_edge("ep-a", "merge")
    dag.add_edge("ep-b", "merge")
    dag.set_input_node("ep-a", "ep-b")
    dag.set_output_node("merge")
    dag.validate()

    input_df = pd.DataFrame({"id": [1, 2], "smiles": ["CCO", "CCN"]})
    out = dag.run(input_df, endpoint_invoker=fake_invoker)

    # The invoker received the input (no Endpoint class involved at all).
    assert set(captured.keys()) == {"ep-a", "ep-b"}
    assert set(captured["ep-a"].columns) == {"id", "smiles"}

    # Concat merged the invoker's outputs.
    assert {"id", "smiles", "out_ep-a", "out_ep-b"}.issubset(out.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
