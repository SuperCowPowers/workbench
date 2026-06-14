"""Tests for the graph-first pipelines_manager: artifact DAG + forward flood.

Freshness is exercised with a fake clock (a dict of ref -> integer "time"), so
no AWS is involved -- mtime_fn just looks the ref up, returning None for refs not
in the dict (i.e. "doesn't exist yet").
"""

import pytest

from workbench.utils.pipelines_manager import (
    Job,
    PipelineGraph,
    parse_spec,
    ref_name,
    ref_type,
    simulated_mtime,
)


def job(script, mode=None, outputs=None, inputs=None, group=None):
    """Build a Job (test convenience)."""
    return Job(script, mode, outputs or [], inputs or [], group)


def clock(times):
    """Fake mtime_fn: look up a ref's time, None if absent."""
    return lambda ref: times.get(ref)


def ran(plan):
    """{node_id: reason} for the jobs a plan would run."""
    return {j.node_id: reason for j, should, reason in plan if should}


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
        jobs = parse_spec(spec)
        assert [j.group for j in jobs] == ["p", "p"]
        assert jobs[0].outputs == ["fs:x"]
        assert jobs[1].mode == "dt"
        assert jobs[1].inputs == ["fs:x"]

    def test_script_resolver_applied(self):
        spec = {"pipelines": {"p": [{"script": "sub/fs.py"}]}}
        jobs = parse_spec(spec, script_resolver=lambda s: f"s3://bucket/{s}")
        assert jobs[0].script == "s3://bucket/sub/fs.py"


class TestConstruction:
    def test_duplicate_producer_raises(self):
        with pytest.raises(ValueError, match="produced by both"):
            PipelineGraph([job("a.py", outputs=["fs:x"]), job("b.py", outputs=["fs:x"])])

    def test_producer_lookup(self):
        fs = job("fs.py", outputs=["fs:x"])
        g = PipelineGraph([fs, job("m.py", inputs=["fs:x"])])
        assert g.producer("fs:x") is fs
        assert g.producer("ds:raw") is None  # external root

    def test_cycle_raises(self):
        a = job("a.py", outputs=["fs:x"], inputs=["fs:y"])
        b = job("b.py", outputs=["fs:y"], inputs=["fs:x"])
        with pytest.raises(ValueError, match="cycle"):
            PipelineGraph([a, b])


class TestArtifactTopology:
    def _chain(self):
        # ds:raw -> fs:x -> model:m
        fs = job("fs.py", outputs=["fs:x"], inputs=["ds:raw"])
        m = job("m.py", outputs=["model:m"], inputs=["fs:x"])
        return PipelineGraph([fs, m])

    def test_ancestors_and_descendants(self):
        g = self._chain()
        assert g.ancestors("model:m") == {"ds:raw", "fs:x"}
        assert g.descendants("ds:raw") == {"fs:x", "model:m"}

    def test_root_can_be_any_type(self):
        # A FeatureSet built elsewhere (no producer) is a perfectly good root.
        m = job("m.py", outputs=["model:m"], inputs=["fs:external"])
        g = PipelineGraph([m])
        assert g.roots() == ["fs:external"]
        assert g.descendants("fs:external") == {"model:m"}

    def test_subdag_upstream(self):
        g = self._chain()
        sub = g.subdag(["model:m"])  # upstream by default
        assert set(sub.nodes) == {"ds:raw", "fs:x", "model:m"}

    def test_subdag_downstream(self):
        g = self._chain()
        sub = g.subdag(["ds:raw"], upstream=False, downstream=True)
        assert set(sub.nodes) == {"ds:raw", "fs:x", "model:m"}


class TestJobOrdering:
    def test_producer_before_consumer(self):
        fs = job("fs.py", outputs=["fs:x"])
        m = job("m.py", inputs=["fs:x"])
        order = PipelineGraph([m, fs]).job_order()
        assert [j.node_id for j in order] == ["fs", "m"]

    def test_cross_pipeline(self):
        fs = job("fs.py", outputs=["fs:x"], group="a")
        m = job("m.py", inputs=["fs:x"], group="b")
        order = PipelineGraph([m, fs]).job_order()
        assert [j.node_id for j in order] == ["fs", "m"]

    def test_subset_still_globally_ordered(self):
        fs = job("fs.py", outputs=["fs:x"])
        m = job("m.py", inputs=["fs:x"], outputs=["model:m"])
        g = PipelineGraph([fs, m])
        assert [j.node_id for j in g.job_order(subset=[m, fs])] == ["fs", "m"]


class TestSelect:
    def test_pulls_transitive_producers(self):
        a = job("a.py", outputs=["fs:x"])
        b = job("b.py", inputs=["fs:x"], outputs=["fs:y"])
        c = job("c.py", inputs=["fs:y"], outputs=["model:c"])
        g = PipelineGraph([a, b, c])
        selected = g.select([c])
        assert {j.script for j in selected} == {"a.py", "b.py", "c.py"}
        assert [j.node_id for j in g.job_order(subset=selected)] == ["a", "b", "c"]

    def test_external_input_adds_nothing(self):
        m = job("m.py", inputs=["ds:raw"])  # ds:raw has no producer
        g = PipelineGraph([m])
        assert {j.script for j in g.select([m])} == {"m.py"}

    def test_unrelated_producers_excluded(self):
        fs = job("fs.py", outputs=["fs:x"])
        m = job("m.py", inputs=["fs:x"])
        other = job("other.py", outputs=["fs:z"])
        g = PipelineGraph([fs, m, other])
        assert {j.script for j in g.select([m])} == {"fs.py", "m.py"}


class TestPlan:
    """Forward flood: a change pushes the full downstream path, in order."""

    def _dag(self):
        # ds:raw -> fs:x (one pipeline) -> model:m (another)
        fs = job("fs.py", outputs=["fs:x"], inputs=["ds:raw"], group="features")
        m = job("m.py", outputs=["model:m"], inputs=["fs:x"], group="models")
        return PipelineGraph([m, fs])  # deliberately out of order

    def test_missing_output_runs(self):
        g = self._dag()
        # model:m absent, fs:x present and fresh -> only m runs (missing).
        plan = g.plan(clock({"ds:raw": 1, "fs:x": 5}))
        assert ran(plan) == {"m": "missing"}

    def test_changed_source_floods_whole_path_in_order(self):
        g = self._dag()
        # ds:raw newer than both outputs: fs goes stale, m inherits via upstream.
        plan = g.plan(clock({"ds:raw": 100, "fs:x": 10, "model:m": 10}))
        scheduled = [j.node_id for j, run, _ in plan if run]
        assert scheduled == ["fs", "m"]
        assert ran(plan) == {"fs": "stale", "m": "upstream"}

    def test_up_to_date_pushes_nothing(self):
        g = self._dag()
        plan = g.plan(clock({"ds:raw": 1, "fs:x": 50, "model:m": 60}))
        assert ran(plan) == {}

    def test_plan_is_topologically_ordered(self):
        g = self._dag()
        order = [j.node_id for j, _, _ in g.plan(clock({"ds:raw": 100}))]
        assert order.index("fs") < order.index("m")

    def test_unmanaged_without_outputs(self):
        g = PipelineGraph([job("x.py", inputs=["ds:raw"])])
        assert ran(g.plan(clock({"ds:raw": 1}))) == {"x": "unmanaged"}

    def test_no_inputs_runs_and_warns(self, capsys):
        g = PipelineGraph([job("x.py", outputs=["model:m"])])
        assert ran(g.plan(clock({"model:m": 1}))) == {"x": "no_inputs"}
        assert "no inputs" in capsys.readouterr().out


class TestMultiOutputJob:
    """A job runs as a unit; any stale output reruns it and refreshes them all."""

    def test_one_stale_output_reruns_job_and_floods_siblings(self):
        # producer emits fs:a + fs:b; consumers of each depend on the one job.
        producer = job("p.py", outputs=["fs:a", "fs:b"], inputs=["ds:raw"])
        ca = job("ca.py", inputs=["fs:a"], outputs=["model:a"])
        cb = job("cb.py", inputs=["fs:b"], outputs=["model:b"])
        g = PipelineGraph([producer, ca, cb])
        # fs:a missing (so p reruns), fs:b + both models present and "fresh".
        plan = g.plan(clock({"ds:raw": 1, "fs:b": 5, "model:a": 5, "model:b": 5}))
        # p reruns (missing fs:a); both consumers inherit even though fs:b looked fresh.
        assert ran(plan) == {"p": "missing", "ca": "upstream", "cb": "upstream"}


class TestSimulatedMtime:
    """Simulating a modified ref propagates forward, whatever the ref's type."""

    def _dag(self):
        fs = job("fs.py", outputs=["fs:x"], inputs=["ds:raw"])
        m = job("m.py", outputs=["model:m"], inputs=["fs:x"])
        return PipelineGraph([fs, m])

    def test_modified_source_propagates(self):
        plan = self._dag().plan(simulated_mtime(["ds:raw"]))
        assert ran(plan) == {"fs": "stale", "m": "upstream"}

    def test_modified_intermediate_fs_now_propagates(self):
        # Forward flood, no backtracking: a modified intermediate FeatureSet
        # triggers its consumers (its own producer stays put -- already fresh).
        plan = self._dag().plan(simulated_mtime(["fs:x"]))
        assert ran(plan) == {"m": "stale"}

    def test_unrelated_source_does_not_trigger(self):
        plan = self._dag().plan(simulated_mtime(["ds:other"]))
        assert ran(plan) == {}

    def test_stale_artifacts_view(self):
        g = self._dag()
        assert g.stale_artifacts(simulated_mtime(["ds:raw"])) == {"fs:x", "model:m"}
