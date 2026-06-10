"""Tests for ml_pipeline_launcher.py - pipelines.json loading, DAG building, and sort_pipelines."""

import json
from pathlib import Path

import pytest

from workbench.scripts.ml_pipeline_launcher import (
    PipelineNode,
    build_dag,
    get_all_pipelines,
    load_pipelines_config,
    parse_script_name,
    run_label,
    sort_pipelines,
)


def node(script, mode=None, outputs=None, inputs=None):
    """Build a PipelineNode with a Path script (test convenience)."""
    return PipelineNode(Path(script), mode, outputs or [], inputs or [])


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


class TestBuildDag:
    """Tests for build_dag artifact-derived edges and validation."""

    def test_edges_derived_from_artifacts(self):
        """An input artifact links the consumer to its producer."""
        producer = node("fs.py", outputs=["fs:x"])
        consumer = node("model.py", inputs=["fs:x"])
        graph = build_dag("p", [producer, consumer])

        assert graph.number_of_edges() == 1
        assert graph.has_edge((Path("fs.py"), None), (Path("model.py"), None))

    def test_fan_out(self):
        """One producer feeding two consumers yields two parallel edges."""
        nodes = [
            node("fs.py", outputs=["fs:x"]),
            node("a.py", inputs=["fs:x"]),
            node("b.py", inputs=["fs:x"]),
        ]
        graph = build_dag("p", nodes)
        assert graph.number_of_edges() == 2
        assert graph.out_degree((Path("fs.py"), None)) == 2

    def test_duplicate_producer_raises(self):
        """Two nodes outputting the same artifact is a hard error."""
        with pytest.raises(ValueError, match="produced by both"):
            build_dag("p", [node("a.py", outputs=["fs:x"]), node("b.py", outputs=["fs:x"])])

    def test_cycle_raises(self):
        """A dependency cycle is a hard error."""
        with pytest.raises(ValueError, match="cycle"):
            build_dag(
                "p",
                [
                    node("a.py", outputs=["fs:x"], inputs=["fs:y"]),
                    node("b.py", outputs=["fs:y"], inputs=["fs:x"]),
                ],
            )

    def test_dangling_input_tolerated(self, capsys):
        """An input with no producer is tolerated silently (it's an external root)."""
        graph = build_dag("p", [node("a.py", inputs=["fs:external"])])
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0
        assert capsys.readouterr().err == ""  # no warning noise

    def test_duplicate_node_raises(self):
        """The same (script, mode) declared twice is a hard error."""
        with pytest.raises(ValueError, match="duplicate node"):
            build_dag("p", [node("a.py", mode="dt"), node("a.py", mode="dt")])


class TestSortPipelines:
    """Tests for sort_pipelines topological ordering and modes."""

    def test_topological_order(self, tmp_path):
        """Producer runs before its consumers."""
        fs = tmp_path / "fs.py"
        a = tmp_path / "a.py"
        dags = {"p": [node(fs, outputs=["fs:x"]), node(a, inputs=["fs:x"])]}

        plan = sort_pipelines([fs, a], dags)

        order = [s for s, _ in plan.runs]
        assert order.index(fs) < order.index(a)
        assert plan.group_ids[(fs, None)] == "p"
        # Artifacts flow straight into the batch dependency tokens
        assert plan.deps[(fs, None)] == {"outputs": ["fs:x"], "inputs": []}
        assert plan.deps[(a, None)] == {"outputs": [], "inputs": ["fs:x"]}

    def test_cross_pipeline_dependency_orders_globally(self, tmp_path):
        """A producer in one pipeline orders before a consumer in another."""
        fs = tmp_path / "fs.py"
        consumer = tmp_path / "consumer.py"
        dags = {
            "producers": [node(fs, outputs=["fs:shared"])],
            "consumers": [node(consumer, inputs=["fs:shared"])],
        }

        plan = sort_pipelines([fs, consumer], dags)

        order = [s for s, _ in plan.runs]
        assert order.index(fs) < order.index(consumer)
        # Each run keeps its own pipeline as its group.
        assert plan.group_ids[(fs, None)] == "producers"
        assert plan.group_ids[(consumer, None)] == "consumers"

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

        plan = sort_pipelines([fs, consumer], dags)
        text = "\n".join(plan.display_lines)
        assert "fs:x" in text  # resolved artifact
        assert "ds:raw" in text  # external root rendered too

    def test_mode_filter_dt(self, tmp_path):
        """Filter mode keeps only matching (and modeless) nodes."""
        xgb = tmp_path / "xgb.py"
        dags = {"p": [node(xgb, "dt", outputs=["fs:x"]), node(xgb, "ts", inputs=["fs:x"])]}

        plan = sort_pipelines([xgb], dags, mode="dt")
        assert plan.runs == [(xgb, "dt")]

    def test_mode_filter_ts_cross_mode_dependency(self, tmp_path):
        """A ts node keeps its input token even when the dt producer is filtered out."""
        xgb = tmp_path / "xgb.py"
        dags = {"p": [node(xgb, "dt", outputs=["fs:x"]), node(xgb, "ts", inputs=["fs:x"])]}

        plan = sort_pipelines([xgb], dags, mode="ts")
        assert plan.runs == [(xgb, "ts")]
        assert plan.deps[(xgb, "ts")]["inputs"] == ["fs:x"]

    def test_no_filter_runs_all_modes(self, tmp_path):
        """With no filter, both modes run, producer before consumer."""
        xgb = tmp_path / "xgb.py"
        dags = {"p": [node(xgb, "dt", outputs=["fs:x"]), node(xgb, "ts", inputs=["fs:x"])]}

        plan = sort_pipelines([xgb], dags)
        assert plan.runs == [(xgb, "dt"), (xgb, "ts")]

    def test_modeless_node_runs_under_filter(self, tmp_path):
        """A modeless node runs under a filter, taking the filter as its runtime mode."""
        a = tmp_path / "a.py"
        plan = sort_pipelines([a], {"p": [node(a)]}, mode="dt")
        assert plan.runs == [(a, "dt")]

    def test_override_mode_dedup(self, tmp_path):
        """Override modes run each unique script once, ignoring node modes and edges."""
        xgb = tmp_path / "xgb.py"
        b = tmp_path / "b.py"
        dags = {"p": [node(xgb, "dt", outputs=["fs:x"]), node(xgb, "ts", inputs=["fs:x"]), node(b, "dt")]}

        plan = sort_pipelines([xgb, b], dags, mode="promote")
        assert plan.runs == [(xgb, "promote"), (b, "promote")]
        assert plan.deps[(xgb, "promote")] == {"outputs": [], "inputs": []}

    def test_filtered_selection_excludes_unselected(self, tmp_path):
        """Only scripts in the selected set appear."""
        fs = tmp_path / "fs.py"
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        dags = {"p": [node(fs, outputs=["fs:x"]), node(a, inputs=["fs:x"]), node(b, inputs=["fs:x"])]}

        plan = sort_pipelines([fs, a], dags)  # b not selected
        scripts = {s for s, _ in plan.runs}
        assert scripts == {fs, a}

    def test_standalone_scripts_included(self, tmp_path):
        """Scripts not in any pipeline appear as standalone runs."""
        a = tmp_path / "a.py"
        orphan = tmp_path / "orphan.py"
        plan = sort_pipelines([a, orphan], {"p": [node(a)]}, mode="dt")

        assert (orphan, "dt") in plan.runs
        assert plan.group_ids[(orphan, "dt")] is None
        assert plan.deps[(orphan, "dt")] == {"outputs": [], "inputs": []}
        assert any("standalone" in line for line in plan.display_lines)

    def test_empty(self):
        """Empty inputs produce an empty plan."""
        plan = sort_pipelines([], {})
        assert plan.runs == []
        assert plan.display_lines == []


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
