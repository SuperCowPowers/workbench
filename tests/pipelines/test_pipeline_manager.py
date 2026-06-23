"""Tests for PipelineManager: discovery, the artifact DAG, and forward-flood scheduling.

Freshness is exercised with a fake clock (a dict of ref -> integer "time"), so
no AWS is involved -- mtime_fn just looks the ref up, returning None for refs not
in the dict (i.e. "doesn't exist yet").
"""

import json
import logging

import pytest

from workbench.lambda_layer.pipeline_manager import (
    Job,
    PipelineManager,
    parse_spec,
    ref_name,
    ref_type,
    simulated_mtime,
)


def job(script, mode=None, outputs=None, inputs=None, pipeline=None):
    """Build a Job (test convenience)."""
    return Job(script, mode, outputs or [], inputs or [], pipeline)


def key(script, mode=None, outputs=None):
    """The Job.key (script, mode, sorted-outputs) for force/blocked assertions."""
    return Job(script, mode, outputs or []).key


def pm(jobs):
    """A PipelineManager over in-memory jobs."""
    return PipelineManager.from_jobs(jobs)


def clock(times):
    """Fake mtime_fn: look up a ref's time, None if absent."""
    return lambda ref: times.get(ref)


def ran(plan):
    """{node_id: reason} for the jobs a plan would run."""
    return {j.node_id: reason for j, should, reason in plan if should}


def arts(graph):
    """The artifact refs in a (bipartite) graph."""
    return {n for n, d in graph.nodes(data=True) if d.get("kind") == "artifact"}


def job_ids(graph):
    """The node_ids of the job nodes in a (bipartite) graph."""
    return {d["job"].node_id for _, d in graph.nodes(data=True) if d.get("kind") == "job"}


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
        assert [j.pipeline for j in jobs] == ["p", "p"]
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
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"], pipeline="features"),
                job("m.py", outputs=["model:m"], inputs=["fs:x"], pipeline="models"),
            ]
        )

    def test_list_and_count(self):
        mgr = self._mgr()
        assert mgr.list_pipelines() == ["features", "models"]
        assert mgr.get_num_pipelines() == 2

    def test_get_pipeline_slice(self):
        mgr = self._mgr()
        slice_ = mgr.get_pipeline("features")
        # The features job plus its own artifacts; its model consumer is elsewhere.
        assert arts(slice_) == {"ds:raw", "fs:x"}
        assert job_ids(slice_) == {"fs"}

    def test_get_pipeline_unknown_raises(self):
        with pytest.raises(KeyError):
            self._mgr().get_pipeline("nope")


class TestDependencyGroupsAPI:
    def _mgr(self):
        # Two disjoint dependency chains -> two dependency groups, regardless of the
        # (human) pipeline labels. The cross-pipeline chain collapses into one group.
        return pm(
            [
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"], pipeline="features"),
                job("m.py", outputs=["model:m"], inputs=["fs:x"], pipeline="models"),
                job("other.py", outputs=["fs:y"], inputs=["ds:other"], pipeline="other"),
            ]
        )

    def test_list_and_count(self):
        mgr = self._mgr()
        # Group id is each chain's root source; fs+model collapse into the ds:raw group.
        assert mgr.list_dependency_groups() == ["ds:raw", "ds:other"]
        assert mgr.get_num_dependency_groups() == 2

    def test_dependency_groups_membership(self):
        groups = self._mgr().dependency_groups()
        assert {gid: sorted(j.node_id for j in jobs) for gid, jobs in groups.items()} == {
            "ds:raw": ["fs", "m"],
            "ds:other": ["other"],
        }

    def test_get_dependency_group_slice(self):
        mgr = self._mgr()
        slice_ = mgr.get_dependency_group("ds:raw")
        # The whole connected component: both jobs and every artifact they touch.
        assert arts(slice_) == {"ds:raw", "fs:x", "model:m"}
        assert job_ids(slice_) == {"fs", "m"}

    def test_get_dependency_group_unknown_raises(self):
        with pytest.raises(KeyError):
            self._mgr().get_dependency_group("nope")


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
        g = self._chain().full_dependency_graph()
        assert arts(g) == {"ds:raw", "fs:x", "model:m"}
        assert job_ids(g) == {"fs", "m"}  # jobs are nodes too (bipartite)

    def test_upstream(self):
        # Upstream of model:m -> the artifacts AND the jobs that produce them.
        g = self._chain().upstream_graph("model:m")
        assert arts(g) == {"ds:raw", "fs:x", "model:m"}
        assert job_ids(g) == {"fs", "m"}

    def test_downstream(self):
        g = self._chain().downstream_graph("ds:raw")
        assert arts(g) == {"ds:raw", "fs:x", "model:m"}
        assert job_ids(g) == {"fs", "m"}

    def test_root_can_be_any_type(self):
        # A FeatureSet built elsewhere (no producer) is a perfectly good root.
        mgr = pm([job("m.py", outputs=["model:m"], inputs=["fs:external"])])
        g = mgr.downstream_graph("fs:external")
        assert arts(g) == {"fs:external", "model:m"}
        assert job_ids(g) == {"m"}


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
        assert "ds:raw" in out  # root artifact
        assert "caco2_fs [dt]" in out  # the producing job is its own node
        assert "fs:caco2" in out  # ...feeding the artifact it produces
        assert "╼" in out  # Unicode tree edges


class TestPlan:
    """Forward flood: a change pushes the full downstream path, in order."""

    def _dag(self):
        # ds:raw -> fs:x -> model:m  (cross-pipeline)
        return pm(
            [
                job("m.py", outputs=["model:m"], inputs=["fs:x"], pipeline="models"),
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"], pipeline="features"),
            ]
        )

    def test_missing_output_runs(self):
        plan = self._dag().plan(clock({"ds:raw": 1, "fs:x": 5}))
        assert ran(plan) == {"m": "missing"}

    def test_changed_source_floods_whole_path_in_order(self):
        plan = self._dag().plan(clock({"ds:raw": 100, "fs:x": 10, "model:m": 10}))
        scheduled = [j.node_id for j, run, _ in plan if run]
        assert scheduled == ["fs", "m"]
        assert ran(plan) == {"fs": "stale", "m": "upstream"}

    def test_up_to_date_pushes_nothing(self):
        plan = self._dag().plan(clock({"ds:raw": 1, "fs:x": 50, "model:m": 60}))
        assert ran(plan) == {}

    def test_defaults_to_builtin_resolver(self):
        # No clock passed -> uses the cached real resolver. Pre-seed the cache so
        # no AWS call happens (every ref in this dag is seeded).
        mgr = self._dag()
        mgr._mtime_cache = {"ds:raw": 100, "fs:x": 10, "model:m": 10}
        assert ran(mgr.plan()) == {"fs": "stale", "m": "upstream"}

    def test_unmanaged_without_outputs(self):
        assert ran(pm([job("x.py", inputs=["ds:raw"])]).plan(clock({"ds:raw": 1}))) == {"x": "unmanaged"}

    def test_no_inputs_runs_and_warns(self, caplog):
        # workbench logger has propagate=False, so attach caplog's handler directly.
        logger = logging.getLogger("workbench")
        logger.addHandler(caplog.handler)
        try:
            plan = pm([job("x.py", outputs=["model:m"])]).plan(clock({"model:m": 1}))
        finally:
            logger.removeHandler(caplog.handler)
        assert ran(plan) == {"x": "no_inputs"}
        assert "no inputs" in caplog.text


class TestMultiOutputJob:
    def test_one_stale_output_reruns_job_and_floods_siblings(self):
        producer = job("p.py", outputs=["fs:a", "fs:b"], inputs=["ds:raw"])
        ca = job("ca.py", inputs=["fs:a"], outputs=["model:a"])
        cb = job("cb.py", inputs=["fs:b"], outputs=["model:b"])
        plan = pm([producer, ca, cb]).plan(clock({"ds:raw": 1, "fs:b": 5, "model:a": 5, "model:b": 5}))
        # fs:a missing -> p reruns; both consumers inherit even though fs:b looked fresh.
        assert ran(plan) == {"p": "missing", "ca": "upstream", "cb": "upstream"}


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


class TestPipelineMeta:
    def test_model_node_carries_names(self):
        meta = job("m.py", mode="dt", outputs=["model:ppb-reg-dt"]).pipeline_meta(serverless=True)
        assert meta == {"serverless": True, "mode": "dt", "model_name": "ppb-reg-dt", "endpoint_name": "ppb-reg-dt"}

    def test_modeless_producer_has_no_model(self):
        meta = job("fs.py", outputs=["fs:x"]).pipeline_meta()
        assert meta == {"serverless": True}


class TestSimulatedMtime:
    def _dag(self):
        return pm(
            [
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"]),
                job("m.py", outputs=["model:m"], inputs=["fs:x"]),
            ]
        )

    def test_modified_source_propagates(self):
        assert ran(self._dag().plan(simulated_mtime(["ds:raw"]))) == {"fs": "stale", "m": "upstream"}

    def test_modified_intermediate_fs_now_propagates(self):
        # Forward flood, no backtracking: a modified intermediate FeatureSet
        # triggers its consumers (its own producer stays put -- already fresh).
        assert ran(self._dag().plan(simulated_mtime(["fs:x"]))) == {"m": "stale"}

    def test_unrelated_source_does_not_trigger(self):
        assert ran(self._dag().plan(simulated_mtime(["ds:other"]))) == {}


class TestForce:
    """Forced job keys run regardless of freshness, and still flood downstream."""

    def _dag(self):
        return pm(
            [
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"]),
                job("m.py", outputs=["model:m"], inputs=["fs:x"]),
            ]
        )

    def test_forced_job_runs_when_up_to_date(self):
        up_to_date = clock({"ds:raw": 1, "fs:x": 5, "model:m": 9})  # nothing stale
        dag = self._dag()
        assert ran(dag.plan(up_to_date)) == {}
        assert ran(dag.plan(up_to_date, force={key("m.py", outputs=["model:m"])})) == {"m": "selected"}

    def test_forced_producer_floods_downstream(self):
        up_to_date = clock({"ds:raw": 1, "fs:x": 5, "model:m": 9})
        assert ran(self._dag().plan(up_to_date, force={key("fs.py", outputs=["fs:x"])})) == {
            "fs": "selected",
            "m": "upstream",
        }


class TestArtifactMtime:
    """Per-ref resolution: a genuinely-absent artifact -> None ("must run"); a
    lookup we couldn't complete (auth/region/throttle) must raise, never None."""

    @staticmethod
    def _fs_client_raising(error):
        class _Client:
            def describe_feature_group(self, **_):
                raise error

        return _Client()

    def test_resource_not_found_means_absent(self):
        from botocore.exceptions import ClientError

        manager = pm([job("x.py", outputs=["fs:foo"])])
        err = ClientError({"Error": {"Code": "ResourceNotFound", "Message": "nope"}}, "DescribeFeatureGroup")
        manager._aws_client = lambda _name: self._fs_client_raising(err)
        assert manager._artifact_mtime("fs:foo") is None

    def test_access_denied_raises(self):
        from botocore.exceptions import ClientError

        manager = pm([job("x.py", outputs=["fs:foo"])])
        err = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "DescribeFeatureGroup")
        manager._aws_client = lambda _name: self._fs_client_raising(err)
        with pytest.raises(ClientError):
            manager._artifact_mtime("fs:foo")

    def test_no_credentials_propagates(self):
        from botocore.exceptions import NoCredentialsError

        manager = pm([job("x.py", outputs=["fs:foo"])])
        manager._aws_client = lambda _name: self._fs_client_raising(NoCredentialsError())
        with pytest.raises(NoCredentialsError):
            manager._artifact_mtime("fs:foo")

    # -- public: (PublicData S3 object via unsigned head_object) --------------

    @staticmethod
    def _public_s3(available):
        """Fake unsigned S3 client: head_object returns LastModified for known keys, 404 otherwise."""
        from botocore.exceptions import ClientError

        class _S3:
            def head_object(self, Bucket, Key):
                if Key in available:
                    return {"LastModified": available[Key]}
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

        return _S3()

    def test_public_resolves_parquet(self):
        manager = pm([job("x.py", inputs=["public:comp_chem/logp/logp_all"])])
        manager._public_s3 = lambda: self._public_s3({"comp_chem/logp/logp_all.parquet": 123})
        assert manager._artifact_mtime("public:comp_chem/logp/logp_all") == 123

    def test_public_falls_back_to_csv(self):
        manager = pm([job("x.py", inputs=["public:foo/bar"])])
        manager._public_s3 = lambda: self._public_s3({"foo/bar.csv": 99})  # no .parquet -> tries .csv
        assert manager._artifact_mtime("public:foo/bar") == 99

    def test_public_absent_returns_none(self):
        manager = pm([job("x.py", inputs=["public:nope/missing"])])
        manager._public_s3 = lambda: self._public_s3({})  # neither extension exists -> must run
        assert manager._artifact_mtime("public:nope/missing") is None


class TestMissingSources:
    """blocked_by_missing_sources: an absent external source dooms a job + its downstream."""

    def test_missing_source_blocks_job_and_downstream(self):
        manager = pm(
            [
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"]),
                job("m.py", inputs=["fs:x"], outputs=["model:m"]),
            ]
        )
        blocked = manager.blocked_by_missing_sources(clock({}))  # ds:raw absent
        fs_key = key("fs.py", outputs=["fs:x"])
        m_key = key("m.py", outputs=["model:m"])
        assert set(blocked) == {fs_key, m_key}
        assert blocked[fs_key] == ["ds:raw"]  # the originating source is reported
        assert blocked[m_key] == ["ds:raw"]  # downstream inherits the cause

    def test_present_source_blocks_nothing(self):
        manager = pm([job("fs.py", outputs=["fs:x"], inputs=["ds:raw"])])
        assert manager.blocked_by_missing_sources(clock({"ds:raw": 1})) == {}

    def test_produced_input_is_not_a_source(self):
        # fs:x has a producer in the DAG -> not a source; its absence doesn't block m.
        manager = pm(
            [
                job("fs.py", outputs=["fs:x"], inputs=["ds:raw"]),
                job("m.py", inputs=["fs:x"], outputs=["model:m"]),
            ]
        )
        assert manager.blocked_by_missing_sources(clock({"ds:raw": 1})) == {}  # fs:x absent, but produced
