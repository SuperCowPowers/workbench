"""Tests for PipelineManager: discovery, the artifact DAG, and forward-flood scheduling.

Freshness is exercised with a fake clock (a dict of ref -> integer "time"), so
no AWS is involved -- mtime_fn just looks the ref up, returning None for refs not
in the dict (i.e. "doesn't exist yet").
"""

import json

import pytest

from workbench.lambda_layer.pipeline_manager import (
    Job,
    PipelineManager,
    parse_spec,
    ref_name,
    ref_type,
    simulated_mtime,
)


def job(script, mode=None, outputs=None, inputs=None, group=None):
    """Build a Job (test convenience)."""
    return Job(script, mode, outputs or [], inputs or [], group)


def pm(jobs):
    """A PipelineManager over in-memory jobs."""
    return PipelineManager.from_jobs(jobs)


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

    def test_script_resolver_applied(self):
        spec = {"pipelines": {"p": [{"script": "sub/fs.py"}]}}
        jobs = parse_spec(spec, script_resolver=lambda s: f"s3://bucket/{s}")
        assert jobs[0].script == "s3://bucket/sub/fs.py"


class TestConstruction:
    def test_duplicate_producer_raises(self):
        with pytest.raises(ValueError, match="produced by both"):
            pm([job("a.py", outputs=["fs:x"]), job("b.py", outputs=["fs:x"])])

    def test_duplicate_job_raises(self):
        with pytest.raises(ValueError, match="duplicate job"):
            pm([job("a.py", mode="dt"), job("a.py", mode="dt")])

    def test_cycle_raises(self):
        a = job("a.py", outputs=["fs:x"], inputs=["fs:y"])
        b = job("b.py", outputs=["fs:y"], inputs=["fs:x"])
        with pytest.raises(ValueError, match="cycle"):
            pm([a, b])


class TestLocalDiscovery:
    def test_discovers_and_resolves_paths(self, tmp_path):
        leaf = tmp_path / "Binding"
        leaf.mkdir()
        config = {
            "pipelines": {
                "ppb": [
                    {"script": "fs.py", "outputs": ["fs:x"]},
                    {"script": "m.py", "mode": "dt", "inputs": ["fs:x"], "outputs": ["model:m"]},
                ]
            }
        }
        (leaf / "pipelines.json").write_text(json.dumps(config))

        mgr = PipelineManager(tmp_path)
        assert mgr.list_pipelines() == ["ppb"]
        # Scripts resolve relative to the config's own directory.
        assert {j.script for j in mgr.jobs} == {leaf / "fs.py", leaf / "m.py"}


class TestPipelinesAPI:
    def _mgr(self):
        return pm(
            [
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"], group="features"),
                job("m.py", outputs=["model:m"], inputs=["fs:x"], group="models"),
            ]
        )

    def test_list_and_count(self):
        mgr = self._mgr()
        assert mgr.list_pipelines() == ["features", "models"]
        assert mgr.get_num_pipelines() == 2

    def test_get_pipeline_slice(self):
        mgr = self._mgr()
        # The features pipeline's own artifacts; its model consumer is elsewhere.
        assert set(mgr.get_pipeline("features").nodes) == {"ds:raw", "fs:x"}

    def test_get_pipeline_unknown_raises(self):
        with pytest.raises(KeyError):
            self._mgr().get_pipeline("nope")


class TestDependencyGraph:
    def _chain(self):
        # ds:raw -> fs:x -> model:m
        return pm(
            [
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"]),
                job("m.py", outputs=["model:m"], inputs=["fs:x"]),
            ]
        )

    def test_full_graph(self):
        assert set(self._chain().full_dependency_graph().nodes) == {"ds:raw", "fs:x", "model:m"}

    def test_upstream(self):
        assert set(self._chain().upstream_graph("model:m").nodes) == {"ds:raw", "fs:x", "model:m"}

    def test_downstream(self):
        assert set(self._chain().downstream_graph("ds:raw").nodes) == {"ds:raw", "fs:x", "model:m"}

    def test_root_can_be_any_type(self):
        # A FeatureSet built elsewhere (no producer) is a perfectly good root.
        mgr = pm([job("m.py", outputs=["model:m"], inputs=["fs:external"])])
        assert set(mgr.downstream_graph("fs:external").nodes) == {"fs:external", "model:m"}


class TestShow:
    def test_renders_refs_and_producers(self, capsys):
        mgr = pm(
            [
                job("caco2_fs.py", mode="dt", outputs=["fs:caco2"], inputs=["ds:raw"]),
                job("caco2_m.py", mode="dt", outputs=["model:caco2"], inputs=["fs:caco2"]),
            ]
        )
        mgr.show(mgr.full_dependency_graph())
        out = capsys.readouterr().out
        assert "ds:raw" in out  # root shows bare
        assert "fs:caco2 <- caco2_fs [dt]" in out  # producer folded into label
        assert "╼" in out  # Unicode tree edges


class TestPlan:
    """Forward flood: a change pushes the full downstream path, in order."""

    def _dag(self):
        # ds:raw -> fs:x -> model:m  (cross-pipeline)
        return pm(
            [
                job("m.py", outputs=["model:m"], inputs=["fs:x"], group="models"),
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"], group="features"),
            ]
        )

    def test_missing_output_runs(self):
        plan = self._dag()._plan(clock({"ds:raw": 1, "fs:x": 5}))
        assert ran(plan) == {"m": "missing"}

    def test_changed_source_floods_whole_path_in_order(self):
        plan = self._dag()._plan(clock({"ds:raw": 100, "fs:x": 10, "model:m": 10}))
        scheduled = [j.node_id for j, run, _ in plan if run]
        assert scheduled == ["fs", "m"]
        assert ran(plan) == {"fs": "stale", "m": "upstream"}

    def test_up_to_date_pushes_nothing(self):
        plan = self._dag()._plan(clock({"ds:raw": 1, "fs:x": 50, "model:m": 60}))
        assert ran(plan) == {}

    def test_unmanaged_without_outputs(self):
        assert ran(pm([job("x.py", inputs=["ds:raw"])])._plan(clock({"ds:raw": 1}))) == {"x": "unmanaged"}

    def test_no_inputs_runs_and_warns(self, capsys):
        plan = pm([job("x.py", outputs=["model:m"])])._plan(clock({"model:m": 1}))
        assert ran(plan) == {"x": "no_inputs"}
        assert "no inputs" in capsys.readouterr().out


class TestMultiOutputJob:
    def test_one_stale_output_reruns_job_and_floods_siblings(self):
        producer = job("p.py", outputs=["fs:a", "fs:b"], inputs=["ds:raw"])
        ca = job("ca.py", inputs=["fs:a"], outputs=["model:a"])
        cb = job("cb.py", inputs=["fs:b"], outputs=["model:b"])
        plan = pm([producer, ca, cb])._plan(clock({"ds:raw": 1, "fs:b": 5, "model:a": 5, "model:b": 5}))
        # fs:a missing -> p reruns; both consumers inherit even though fs:b looked fresh.
        assert ran(plan) == {"p": "missing", "ca": "upstream", "cb": "upstream"}


class TestOrderedBatchJobs:
    def _dag(self):
        return pm(
            [
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"]),
                job("m.py", outputs=["model:m"], inputs=["fs:x"]),
            ]
        )

    def test_all_jobs_when_no_clock(self):
        assert [j.node_id for j in self._dag()._ordered_batch_jobs()] == ["fs", "m"]

    def test_only_stale_with_clock(self):
        jobs = self._dag()._ordered_batch_jobs(clock({"ds:raw": 1, "fs:x": 50, "model:m": 60}))
        assert jobs == []  # all up to date

    def test_subset_restricts_but_keeps_order(self):
        mgr = self._dag()
        only_fs = [j for j in mgr.jobs if j.stem == "fs"]
        assert [j.node_id for j in mgr._ordered_batch_jobs(subset=only_fs)] == ["fs"]


class TestSelect:
    def test_pulls_transitive_producers(self):
        a = job("a.py", outputs=["fs:x"])
        b = job("b.py", inputs=["fs:x"], outputs=["fs:y"])
        c = job("c.py", inputs=["fs:y"], outputs=["model:c"])
        mgr = pm([a, b, c])
        assert {j.script for j in mgr._select([c])} == {"a.py", "b.py", "c.py"}

    def test_external_input_adds_nothing(self):
        m = job("m.py", inputs=["ds:raw"])
        assert {j.script for j in pm([m])._select([m])} == {"m.py"}


class TestSimulatedMtime:
    def _dag(self):
        return pm(
            [
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"]),
                job("m.py", outputs=["model:m"], inputs=["fs:x"]),
            ]
        )

    def test_modified_source_propagates(self):
        assert ran(self._dag()._plan(simulated_mtime(["ds:raw"]))) == {"fs": "stale", "m": "upstream"}

    def test_modified_intermediate_fs_now_propagates(self):
        # Forward flood, no backtracking: a modified intermediate FeatureSet
        # triggers its consumers (its own producer stays put -- already fresh).
        assert ran(self._dag()._plan(simulated_mtime(["fs:x"]))) == {"m": "stale"}

    def test_unrelated_source_does_not_trigger(self):
        assert ran(self._dag()._plan(simulated_mtime(["ds:other"]))) == {}
