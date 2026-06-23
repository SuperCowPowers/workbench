"""Tests for ml_pipeline_launcher.py - pipelines.json loading, DAG building, and sort_pipelines."""

import json
from pathlib import Path

import pytest

from workbench.scripts.ml_pipeline_launcher import (
    Job,
    build_pipeline_meta,
    get_all_pipelines,
    load_pipelines_config,
    parse_script_name,
    run_label,
    run_simulation,
    sort_pipelines,
)
from workbench.lambda_layer.pipeline_manager import PipelineManager


def node(script, mode=None, outputs=None, inputs=None):
    """Build a Job with a Path script (test convenience)."""
    return Job(Path(script), mode, outputs or [], inputs or [])


def run_keys(plan):
    """[(script, mode), ...] for the plan's run-jobs."""
    return [(j.script, j.mode) for j in plan.runs]


def run_by(plan, script, mode):
    """The run-job in the plan matching (script, mode)."""
    return next(j for j in plan.runs if j.script == script and j.mode == mode)


@pytest.fixture(autouse=True)
def all_stale(monkeypatch):
    """Launcher tests run without AWS. Model a genuine from-scratch build: raw
    sources (ds:/public:) exist, but every derived artifact (fs:/model:) is absent,
    so all selected jobs are stale and run. These tests cover selection / ordering /
    mode / display -- not freshness (test_pipeline_manager) or missing-source pruning
    (TestMissingSourceSkip), both of which override this resolver.
    """
    monkeypatch.setattr(
        PipelineManager,
        "_artifact_mtime",
        lambda self, ref: 1 if ref.startswith(("ds:", "public:")) else None,
    )


class TestLoadPipelinesConfig:
    """Tests for loading pipelines.json into PipelineNode lists."""

    def test_load_valid_json(self, tmp_path):
        """Load a valid pipelines.json with outputs/inputs nodes."""
        config = {
            "pipelines": {
                "aqsol": [
                    {"script": "aqsol_feature_set.py", "outputs": ["fs:aqsol_features"]},
                    {"script": "aqsol_class.py", "inputs": ["fs:aqsol_features"]},
                ]
            }
        }
        (tmp_path / "pipelines.json").write_text(json.dumps(config))
        result = load_pipelines_config(tmp_path)

        assert result is not None
        assert list(result) == ["aqsol"]
        nodes = result["aqsol"]
        assert len(nodes) == 2
        assert nodes[0].script == tmp_path / "aqsol_feature_set.py"
        assert nodes[0].outputs == ["fs:aqsol_features"]
        assert nodes[0].inputs == []
        assert nodes[1].inputs == ["fs:aqsol_features"]

    def test_modeless_and_moded_nodes(self, tmp_path):
        """Nodes may omit mode (modeless) or declare a singular mode."""
        config = {
            "pipelines": {
                "p": [
                    {"script": "a.py"},
                    {"script": "b.py", "mode": "dt", "outputs": ["fs:x"]},
                    {"script": "b.py", "mode": "ts", "inputs": ["fs:x"]},
                ]
            }
        }
        (tmp_path / "pipelines.json").write_text(json.dumps(config))
        nodes = load_pipelines_config(tmp_path)["p"]
        assert nodes[0].mode is None
        assert nodes[1].mode == "dt"
        assert nodes[2].mode == "ts"

    def test_returns_none_when_no_json(self, tmp_path):
        """Returns None when no pipelines.json exists in the directory."""
        assert load_pipelines_config(tmp_path) is None

    def test_empty_pipelines(self, tmp_path):
        """Handle JSON with an empty pipelines section."""
        (tmp_path / "pipelines.json").write_text(json.dumps({"pipelines": {}}))
        assert load_pipelines_config(tmp_path) == {}

    def test_paths_resolved_relative_to_directory(self, tmp_path):
        """Script paths should be resolved relative to the JSON config's directory."""
        config = {"pipelines": {"p": [{"script": "subdir_script.py"}]}}
        (tmp_path / "pipelines.json").write_text(json.dumps(config))
        nodes = load_pipelines_config(tmp_path)["p"]
        assert nodes[0].script == tmp_path / "subdir_script.py"


class TestJobGraph:
    """Tests for the bipartite dependency DAG the launcher orders on (PipelineManager.graph)."""

    def test_edges_derived_from_artifacts(self):
        """Producer and consumer are linked *through* the artifact node (bipartite)."""
        producer = node("fs.py", outputs=["fs:x"])
        consumer = node("model.py", inputs=["fs:x"])
        graph = PipelineManager.from_jobs([producer, consumer]).graph

        assert graph.has_edge(producer.key, "fs:x")  # job -> artifact (produces)
        assert graph.has_edge("fs:x", consumer.key)  # artifact -> job (consumes)

    def test_fan_out(self):
        """One artifact feeding two consumers yields two edges out of the artifact."""
        fs_node = node("fs.py", outputs=["fs:x"])
        nodes = [
            fs_node,
            node("a.py", inputs=["fs:x"]),
            node("b.py", inputs=["fs:x"]),
        ]
        graph = PipelineManager.from_jobs(nodes).graph
        assert graph.out_degree("fs:x") == 2  # feeds a.py and b.py
        assert graph.out_degree(fs_node.key) == 1  # produces just fs:x

    def test_dangling_input_tolerated(self, capsys):
        """An input with no producer is tolerated silently (it's an external root)."""
        a_node = node("a.py", inputs=["fs:external"])
        graph = PipelineManager.from_jobs([a_node]).graph
        assert graph.number_of_nodes() == 2  # the job + its external input artifact
        assert graph.has_edge("fs:external", a_node.key)
        assert capsys.readouterr().err == ""  # no warning noise

    # Note: duplicate-producer / cycle / duplicate-job validation is covered by
    # TestConstruction in test_pipeline_manager.py (graph-building is a manager concern).


class TestSortPipelines:
    """Tests for sort_pipelines topological ordering and modes."""

    def test_topological_order(self, tmp_path):
        """Producer runs before its consumers."""
        fs = tmp_path / "fs.py"
        a = tmp_path / "a.py"
        dags = {"p": [node(fs, outputs=["fs:x"]), node(a, inputs=["fs:x"])]}

        plan = sort_pipelines([fs, a], dags)

        order = [j.script for j in plan.runs]
        assert order.index(fs) < order.index(a)
        assert run_by(plan, fs, None).pipeline == "p"
        # Artifacts flow straight into the batch dependency tokens
        assert (run_by(plan, fs, None).outputs, run_by(plan, fs, None).inputs) == (["fs:x"], [])
        assert (run_by(plan, a, None).outputs, run_by(plan, a, None).inputs) == ([], ["fs:x"])

    def test_cross_pipeline_dependency_orders_globally(self, tmp_path):
        """A producer in one pipeline orders before a consumer in another."""
        fs = tmp_path / "fs.py"
        consumer = tmp_path / "consumer.py"
        dags = {
            "producers": [node(fs, outputs=["fs:shared"])],
            "consumers": [node(consumer, inputs=["fs:shared"])],
        }

        plan = sort_pipelines([fs, consumer], dags)

        order = [j.script for j in plan.runs]
        assert order.index(fs) < order.index(consumer)
        # Each run keeps its own pipeline name (the human grouping)...
        assert run_by(plan, fs, None).pipeline == "producers"
        assert run_by(plan, consumer, None).pipeline == "consumers"
        # ...but producer and consumer share one dependency group -> one SQS FIFO group, so
        # the queue drains them in topological order and the consumer's dependsOn resolves.
        assert run_by(plan, fs, None).group == run_by(plan, consumer, None).group

    def test_external_inputs_tolerated_silently(self, tmp_path, capsys):
        """Dangling/external inputs emit no warnings -- the render's grey/green conveys them."""
        fs = tmp_path / "fs.py"
        consumer = tmp_path / "consumer.py"
        dags = {
            "producers": [node(fs, outputs=["fs:shared"])],
            "consumers": [node(consumer, inputs=["fs:shared", "ds:raw", "fs:missing"])],
        }

        sort_pipelines([fs, consumer], dags)  # all selected
        sort_pipelines([consumer], dags)  # producer NOT selected -> fs:shared external here
        assert "external input" not in capsys.readouterr().err

    def test_dataflow_display(self, tmp_path):
        """Display renders the data-flow DAG: artifacts as nodes, producer above consumers."""
        producer = tmp_path / "producer.py"
        cons_a = tmp_path / "cons_a.py"
        cons_b = tmp_path / "cons_b.py"
        dags = {
            "p": [
                node(producer, outputs=["fs:x"]),
                node(cons_a, inputs=["fs:x"]),
                node(cons_b, inputs=["fs:x"]),
            ]
        }

        plan = sort_pipelines([producer, cons_a, cons_b], dags)

        text = "\n".join(plan.display_lines)
        assert "fs:x" in text  # the artifact is rendered as a node
        assert any("╼" in line for line in plan.display_lines)  # Unicode tree edges
        # Producer script -> the artifact it makes -> the consumers.
        assert text.index("producer") < text.index("fs:x") < text.index("cons_a")

    def test_external_input_greyed(self, tmp_path):
        """An input with no producer renders (it's a node); resolved + external both show."""
        fs = tmp_path / "fs.py"
        consumer = tmp_path / "consumer.py"
        dags = {"p": [node(fs, outputs=["fs:x"]), node(consumer, inputs=["fs:x", "ds:raw"])]}

        plan = sort_pipelines([fs, consumer], dags)  # ds:raw present (from-scratch fixture)
        text = "\n".join(plan.display_lines)
        assert "fs:x" in text  # resolved artifact
        assert "ds:raw" in text  # external root rendered too

    def test_mode_filter_dt(self, tmp_path):
        """Filter mode keeps only matching (and modeless) nodes."""
        xgb = tmp_path / "xgb.py"
        dags = {"p": [node(xgb, "dt", outputs=["fs:x"]), node(xgb, "ts", inputs=["fs:x"])]}

        plan = sort_pipelines([xgb], dags, mode="dt")
        assert run_keys(plan) == [(xgb, "dt")]

    def test_mode_filter_ts_cross_mode_dependency(self, tmp_path):
        """A ts node keeps its input token even when the dt producer is filtered out."""
        xgb = tmp_path / "xgb.py"
        dags = {"p": [node(xgb, "dt", outputs=["fs:x"]), node(xgb, "ts", inputs=["fs:x"])]}

        plan = sort_pipelines([xgb], dags, mode="ts")
        assert run_keys(plan) == [(xgb, "ts")]
        assert run_by(plan, xgb, "ts").inputs == ["fs:x"]

    def test_no_filter_runs_all_modes(self, tmp_path):
        """With no filter, both modes run, producer before consumer."""
        xgb = tmp_path / "xgb.py"
        dags = {"p": [node(xgb, "dt", outputs=["fs:x"]), node(xgb, "ts", inputs=["fs:x"])]}

        plan = sort_pipelines([xgb], dags)
        assert run_keys(plan) == [(xgb, "dt"), (xgb, "ts")]

    def test_modeless_node_runs_under_filter(self, tmp_path):
        """A modeless node runs under a filter, taking the filter as its runtime mode."""
        a = tmp_path / "a.py"
        plan = sort_pipelines([a], {"p": [node(a)]}, mode="dt")
        assert run_keys(plan) == [(a, "dt")]

    def test_override_mode_dedup(self, tmp_path):
        """Override modes run each unique script once, ignoring node modes and edges."""
        xgb = tmp_path / "xgb.py"
        b = tmp_path / "b.py"
        dags = {"p": [node(xgb, "dt", outputs=["fs:x"]), node(xgb, "ts", inputs=["fs:x"]), node(b, "dt")]}

        plan = sort_pipelines([xgb, b], dags, mode="promote")
        assert run_keys(plan) == [(xgb, "promote"), (b, "promote")]
        assert (run_by(plan, xgb, "promote").outputs, run_by(plan, xgb, "promote").inputs) == ([], [])

    def test_filtered_selection_excludes_unselected(self, tmp_path):
        """Only scripts in the selected set appear."""
        fs = tmp_path / "fs.py"
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        dags = {"p": [node(fs, outputs=["fs:x"]), node(a, inputs=["fs:x"]), node(b, inputs=["fs:x"])]}

        plan = sort_pipelines([fs, a], dags)  # b not selected
        scripts = {j.script for j in plan.runs}
        assert scripts == {fs, a}

    def test_standalone_scripts_included(self, tmp_path):
        """Scripts not in any pipeline appear as standalone runs."""
        a = tmp_path / "a.py"
        orphan = tmp_path / "orphan.py"
        plan = sort_pipelines([a, orphan], {"p": [node(a)]}, mode="dt")

        assert (orphan, "dt") in run_keys(plan)
        assert run_by(plan, orphan, "dt").pipeline is None  # standalone: no pipeline, no dependency group
        assert (run_by(plan, orphan, "dt").outputs, run_by(plan, orphan, "dt").inputs) == ([], [])
        assert any("standalone" in line for line in plan.display_lines)

    def test_empty(self):
        """Empty inputs produce an empty plan."""
        plan = sort_pipelines([], {})
        assert plan.runs == []
        assert plan.display_lines == []


class TestFreshness:
    """The launcher is mtime-aware: only stale jobs run; --full-dag still shows all."""

    def _dags(self, tmp_path):
        fs = tmp_path / "fs.py"
        m = tmp_path / "m.py"
        dags = {"p": [node(fs, outputs=["fs:x"], inputs=["ds:raw"]), node(m, inputs=["fs:x"], outputs=["model:m"])]}
        return fs, m, dags

    def test_up_to_date_runs_nothing(self, tmp_path, monkeypatch):
        fs, m, dags = self._dags(tmp_path)
        times = {"ds:raw": 1, "fs:x": 50, "model:m": 60}  # outputs newer than source
        monkeypatch.setattr(PipelineManager, "_artifact_mtime", lambda self, ref: times.get(ref))
        assert sort_pipelines([fs, m], dags).runs == []

    def test_stale_source_runs_whole_path(self, tmp_path, monkeypatch):
        fs, m, dags = self._dags(tmp_path)
        times = {"ds:raw": 100, "fs:x": 10, "model:m": 10}  # source newer -> both stale
        monkeypatch.setattr(PipelineManager, "_artifact_mtime", lambda self, ref: times.get(ref))
        assert {j.script for j in sort_pipelines([fs, m], dags).runs} == {fs, m}

    def test_full_dag_shows_closure_even_when_nothing_stale(self, tmp_path, monkeypatch):
        fs, m, dags = self._dags(tmp_path)
        times = {"ds:raw": 1, "fs:x": 50, "model:m": 60}
        monkeypatch.setattr(PipelineManager, "_artifact_mtime", lambda self, ref: times.get(ref))
        plan = sort_pipelines([fs, m], dags, full_dag=True)
        assert plan.runs == []  # nothing stale -> nothing submits
        text = "\n".join(plan.display_lines)
        assert "fs:x" in text and "model:m" in text  # ...but the closure is still rendered

    def test_force_runs_matched_script_when_up_to_date(self, tmp_path, monkeypatch):
        # User named a pattern -> the matched script runs even though its deps are current.
        fs, m, dags = self._dags(tmp_path)
        times = {"ds:raw": 1, "fs:x": 50, "model:m": 60}
        monkeypatch.setattr(PipelineManager, "_artifact_mtime", lambda self, ref: times.get(ref))
        plan = sort_pipelines([fs, m], dags, force_keys={dags["p"][1].key})
        assert {j.script for j in plan.runs} == {m}  # only the forced match; up-to-date fs stays put

    def _suffix_text(self, tmp_path, monkeypatch, times):
        fs, m, dags = self._dags(tmp_path)
        monkeypatch.setattr(PipelineManager, "_artifact_mtime", lambda self, ref: times.get(ref))
        plan = sort_pipelines([fs, m], dags, full_dag=True)  # full_dag so nodes show even when nothing runs
        return "\n".join(plan.display_lines)

    def test_dep_suffix_current(self, tmp_path, monkeypatch):
        text = self._suffix_text(tmp_path, monkeypatch, {"ds:raw": 1, "fs:x": 5, "model:m": 9})
        assert "fs:x (current)" in text  # fs older than the model built from it

    def test_dep_suffix_modified(self, tmp_path, monkeypatch):
        text = self._suffix_text(tmp_path, monkeypatch, {"ds:raw": 1, "fs:x": 20, "model:m": 9})
        assert "fs:x (modified)" in text  # fs newer than the model -> would trigger a rebuild

    def test_dep_suffix_missing(self, tmp_path, monkeypatch):
        text = self._suffix_text(tmp_path, monkeypatch, {"ds:raw": 1, "model:m": 9})  # fs:x absent
        assert "fs:x (missing)" in text

    def test_run_reason_on_script_node(self, tmp_path, monkeypatch):
        # model:m absent -> the m script runs as (missing); fs is current.
        fs, m, dags = self._dags(tmp_path)
        times = {"ds:raw": 1, "fs:x": 50}  # model:m absent
        monkeypatch.setattr(PipelineManager, "_artifact_mtime", lambda self, ref: times.get(ref))
        text = "\n".join(sort_pipelines([fs, m], dags).display_lines)
        assert "m (missing)" in text  # the run reason is shown on the script node


class TestMissingSourceSkip:
    """A job whose external source input is missing is pruned (with its downstream)."""

    def _dags(self, tmp_path):
        fs = tmp_path / "fs.py"
        m = tmp_path / "m.py"
        dags = {"p": [node(fs, outputs=["fs:x"], inputs=["ds:raw"]), node(m, inputs=["fs:x"], outputs=["model:m"])]}
        return fs, m, dags

    def test_missing_source_skips_job_and_downstream(self, tmp_path, monkeypatch):
        # ds:raw absent -> fs is doomed, and m (downstream) with it.
        fs, m, dags = self._dags(tmp_path)
        monkeypatch.setattr(PipelineManager, "_artifact_mtime", lambda self, ref: None)
        plan = sort_pipelines([fs, m], dags)
        assert plan.runs == []
        assert {j.script for j, _ in plan.skipped} == {fs, m}
        assert all("ds:raw" in refs for _, refs in plan.skipped)  # the originating source is reported

    def test_present_source_runs_normally(self, tmp_path, monkeypatch):
        fs, m, dags = self._dags(tmp_path)
        monkeypatch.setattr(PipelineManager, "_artifact_mtime", lambda self, ref: 1 if ref == "ds:raw" else None)
        plan = sort_pipelines([fs, m], dags)
        assert {j.script for j in plan.runs} == {fs, m}
        assert plan.skipped == []

    def test_healthy_sibling_still_runs(self, tmp_path, monkeypatch):
        # Pipeline a's source is missing; b's is present -> only a is pruned.
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        dags = {
            "a": [node(a, inputs=["ds:a_raw"], outputs=["model:a"])],
            "b": [node(b, inputs=["ds:b_raw"], outputs=["model:b"])],
        }
        monkeypatch.setattr(PipelineManager, "_artifact_mtime", lambda self, ref: 1 if ref == "ds:b_raw" else None)
        plan = sort_pipelines([a, b], dags)
        assert {j.script for j in plan.runs} == {b}
        assert {j.script for j, _ in plan.skipped} == {a}


class TestRunSimulation:
    """--sim-mod: offline forward-flood view of what a modified ref would submit."""

    def test_renders_triggered_path_and_always_run(self, capsys):
        # A no-inputs job is "always-run"; a ds -> fs chain is triggered by the sim.
        dags = {
            "p": [
                node("logp.py", "dt", outputs=["model:logp-dt"]),  # no inputs -> always-run
                node("fs.py", outputs=["fs:x"], inputs=["ds:raw"]),
            ]
        }
        run_simulation(dags, ["ds:raw"])  # regression: must not raise on the always-run branch
        out = capsys.readouterr().out
        assert "ds:raw" in out  # the triggered path is rendered
        assert "Always-run regardless" in out  # the no-inputs job is listed separately
        assert "logp [dt]" in out


class TestGetAllPipelines:
    """Tests for get_all_pipelines discovery."""

    def test_nested_json_discovered(self, tmp_path, monkeypatch):
        """pipelines.json in a nested directory is found, with standalone scripts."""
        leaf = tmp_path / "top" / "leaf"
        leaf.mkdir(parents=True)
        (leaf / "fs.py").write_text("# fs\n")
        (leaf / "model.py").write_text("# model\n")
        (leaf / "standalone.py").write_text("# standalone\n")
        config = {
            "pipelines": {
                "p": [
                    {"script": "fs.py", "outputs": ["fs:x"]},
                    {"script": "model.py", "inputs": ["fs:x"]},
                ]
            }
        }
        (leaf / "pipelines.json").write_text(json.dumps(config))

        monkeypatch.chdir(tmp_path)
        pipelines, all_dags = get_all_pipelines()

        names = {p.name for p in pipelines}
        assert {"fs.py", "model.py", "standalone.py"} <= names
        assert "p" in all_dags

    def test_dunder_files_excluded(self, tmp_path, monkeypatch):
        """__init__.py and other dunder files are excluded."""
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / "real_script.py").write_text("# real\n")

        monkeypatch.chdir(tmp_path)
        pipelines, _ = get_all_pipelines()

        names = {p.name for p in pipelines}
        assert "real_script.py" in names
        assert "__init__.py" not in names


class TestRunLabel:
    """Tests for the run_label helper."""

    def test_with_mode(self):
        assert run_label(Path("foo/xgb.py"), "dt") == "xgb [dt]"

    def test_without_mode(self):
        assert run_label(Path("foo/xgb.py"), None) == "xgb"


class TestParseScriptName:
    """Tests for parse_script_name filename parsing with optional version."""

    def test_standard_version(self):
        assert parse_script_name(Path("my_script_1.py")) == ("my_script", "1")

    def test_v_prefix_version(self):
        assert parse_script_name(Path("my_script_v2.py")) == ("my_script", "2")

    def test_multi_digit_version(self):
        assert parse_script_name(Path("my_script_12.py")) == ("my_script", "12")

    def test_real_pipeline_filename(self):
        assert parse_script_name(Path("ppb_human_free_reg_xgb_1.py")) == ("ppb_human_free_reg_xgb", "1")

    def test_no_version_returns_none(self):
        assert parse_script_name(Path("caco2_er_reg_open_admet.py")) == ("caco2_er_reg_open_admet", None)

    def test_non_numeric_suffix_returns_none(self):
        assert parse_script_name(Path("my_script_abc.py")) == ("my_script_abc", None)


class TestBuildPipelineMeta:
    """build_pipeline_meta prefers the declared model: output, falls back to the filename."""

    def test_dt_uses_declared_model_ref(self):
        meta = json.loads(
            build_pipeline_meta(node("ppb_human_free_reg_xgb_1.py", "dt", outputs=["model:custom-name"]), True)
        )
        assert meta["model_name"] == "custom-name"
        assert meta["endpoint_name"] == "custom-name"

    def test_dt_falls_back_to_filename(self):
        meta = json.loads(build_pipeline_meta(node("ppb_human_free_reg_xgb_1.py", "dt"), True))
        assert meta["model_name"] == "ppb-human-free-reg-xgb-1-dt"

    def test_modeless_has_no_model(self):
        meta = json.loads(build_pipeline_meta(node("ppb_human_feature_sets_1.py", None), True))
        assert "model_name" not in meta
        assert meta["serverless"] is True
