"""Tests for the canonical pipelines_manager: graph wiring and freshness.

Freshness is exercised with a fake clock (a dict of ref -> integer "time"), so
no AWS is involved -- mtime_fn just looks the ref up, returning None for refs not
in the dict (i.e. "doesn't exist yet").
"""

import pytest

from workbench.utils.pipelines_manager import (
    PipelineNode,
    build_producer_index,
    derive_edges,
    effective_source_time,
    needs_run,
    parse_spec,
    plan_runs,
    ref_name,
    ref_type,
    topo_order,
)


def node(script, mode=None, outputs=None, inputs=None, group=None):
    """Build a PipelineNode (test convenience)."""
    return PipelineNode(script, mode, outputs or [], inputs or [], group)


def clock(times):
    """Fake mtime_fn: look up a ref's time, None if absent."""
    return lambda ref: times.get(ref)


class TestRefHelpers:
    def test_type_and_name(self):
        assert ref_type("fs:caco2_1") == "fs"
        assert ref_name("fs:caco2_1") == "caco2_1"

    def test_model_ref(self):
        assert ref_type("model:ppb-human-1-dt") == "model"
        assert ref_name("model:ppb-human-1-dt") == "ppb-human-1-dt"


class TestParseSpec:
    def test_groups_and_fields(self):
        spec = {
            "pipelines": {
                "p": [
                    {"script": "fs.py", "outputs": ["fs:x"]},
                    {"script": "m.py", "mode": "dt", "inputs": ["fs:x"], "outputs": ["model:m-dt"]},
                ]
            }
        }
        nodes = parse_spec(spec)
        assert [n.group for n in nodes] == ["p", "p"]
        assert nodes[0].outputs == ["fs:x"]
        assert nodes[1].mode == "dt"
        assert nodes[1].inputs == ["fs:x"]

    def test_script_resolver_applied(self):
        spec = {"pipelines": {"p": [{"script": "sub/fs.py"}]}}
        nodes = parse_spec(spec, script_resolver=lambda s: f"s3://bucket/{s}")
        assert nodes[0].script == "s3://bucket/sub/fs.py"


class TestProducerIndexAndEdges:
    def test_index_maps_output_to_node(self):
        fs = node("fs.py", outputs=["fs:x"])
        index = build_producer_index([fs, node("m.py", inputs=["fs:x"])])
        assert index["fs:x"] is fs

    def test_duplicate_producer_raises(self):
        with pytest.raises(ValueError, match="produced by both"):
            build_producer_index([node("a.py", outputs=["fs:x"]), node("b.py", outputs=["fs:x"])])

    def test_edges_and_externals(self):
        fs = node("fs.py", outputs=["fs:x"])
        consumer = node("m.py", inputs=["fs:x", "ds:raw"])
        nodes = [fs, consumer]
        edges, externals = derive_edges(nodes, build_producer_index(nodes))
        assert edges == [(fs, consumer)]
        assert externals == [(consumer, "ds:raw")]


class TestTopoOrder:
    def test_producer_before_consumer(self):
        fs = node("fs.py", outputs=["fs:x"])
        m = node("m.py", inputs=["fs:x"])
        assert topo_order([m, fs]) == [fs, m]

    def test_cross_pipeline(self):
        """A producer in one pipeline orders before a consumer in another."""
        fs = node("fs.py", outputs=["fs:x"], group="a")
        m = node("m.py", inputs=["fs:x"], group="b")
        assert topo_order([m, fs]) == [fs, m]

    def test_cycle_raises(self):
        a = node("a.py", outputs=["fs:x"], inputs=["fs:y"])
        b = node("b.py", outputs=["fs:y"], inputs=["fs:x"])
        with pytest.raises(ValueError, match="cycle"):
            topo_order([a, b])


class TestEffectiveSourceTime:
    def test_walks_to_leaf_datasource(self):
        # ds:raw -> fs:x -> model:m ; freshness of model roots at ds:raw's time.
        fs = node("fs.py", outputs=["fs:x"], inputs=["ds:raw"])
        m = node("m.py", outputs=["model:m"], inputs=["fs:x"])
        index = build_producer_index([fs, m])
        assert effective_source_time("fs:x", index, clock({"ds:raw": 100})) == 100

    def test_takes_latest_of_multiple_inputs(self):
        fs = node("fs.py", outputs=["fs:x"], inputs=["ds:a", "ds:b"])
        index = build_producer_index([fs])
        assert effective_source_time("fs:x", index, clock({"ds:a": 5, "ds:b": 9})) == 9

    def test_leaf_uses_own_time(self):
        # No producer for ds:raw -> use its own mtime.
        assert effective_source_time("ds:raw", {}, clock({"ds:raw": 42})) == 42


class TestNeedsRun:
    def _graph(self):
        fs = node("fs.py", outputs=["fs:x"], inputs=["ds:raw"])
        m = node("m.py", outputs=["model:m"], inputs=["fs:x"])
        return m, build_producer_index([fs, m])

    def test_missing_output(self):
        m, index = self._graph()
        # model:m absent -> must run.
        assert needs_run(m, index, clock({"ds:raw": 1})) == (True, "missing")

    def test_stale_when_source_newer(self):
        m, index = self._graph()
        # source (ds:raw=10) newer than output (model:m=5) -> stale.
        assert needs_run(m, index, clock({"ds:raw": 10, "model:m": 5})) == (True, "stale")

    def test_up_to_date(self):
        m, index = self._graph()
        # output (20) newer than source (10) -> skip.
        assert needs_run(m, index, clock({"ds:raw": 10, "model:m": 20})) == (False, "up_to_date")

    def test_unmanaged_without_outputs(self):
        n = node("x.py", inputs=["ds:raw"])
        assert needs_run(n, {}, clock({"ds:raw": 1})) == (True, "unmanaged")

    def test_no_inputs_runs(self, capsys):
        n = node("x.py", outputs=["model:m"])
        assert needs_run(n, {}, clock({"model:m": 1})) == (True, "no_inputs")
        assert "no inputs" in capsys.readouterr().out


class TestPlanRuns:
    """The scheduler: source change pushes the full downstream path, in order."""

    def _dag(self):
        # ds:raw -> fs:x (cross to another pipeline) -> model:m
        fs = node("fs.py", outputs=["fs:x"], inputs=["ds:raw"], group="features")
        m = node("m.py", outputs=["model:m"], inputs=["fs:x"], group="models")
        return [m, fs]  # deliberately out of order

    def test_changed_source_pushes_whole_path_in_order(self):
        nodes = self._dag()
        # ds:raw newer than both outputs -> both stale, producer first.
        plan = plan_runs(nodes, clock({"ds:raw": 100, "fs:x": 10, "model:m": 10}))
        # Producer (fs) ordered before consumer (m), both flagged stale.
        scheduled = [n.node_id for n, run, _ in plan if run]
        assert scheduled == ["fs", "m"]
        assert all(reason == "stale" for _, run, reason in plan if run)

    def test_up_to_date_source_pushes_nothing(self):
        nodes = self._dag()
        # outputs newer than source -> nothing to push.
        plan = plan_runs(nodes, clock({"ds:raw": 1, "fs:x": 50, "model:m": 60}))
        assert [n.node_id for n, run, _ in plan if run] == []

    def test_plan_is_topologically_ordered(self):
        nodes = self._dag()
        plan = plan_runs(nodes, clock({"ds:raw": 100}))  # everything missing/stale
        order = [n.node_id for n, _, _ in plan]
        assert order.index("fs") < order.index("m")
