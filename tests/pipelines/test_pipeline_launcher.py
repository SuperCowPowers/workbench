"""Tests for ml_pipeline_launcher.py - pipelines.json loading and sort_pipelines."""

import json
import os
from unittest.mock import patch

from workbench.scripts.ml_pipeline_launcher import (
    get_all_pipelines,
    load_pipelines_config,
    parse_script_name,
    sort_pipelines,
)


class TestLoadPipelinesConfig:
    """Tests for loading pipelines.json with DAG format."""

    def test_load_valid_json(self, tmp_path):
        """Load a valid pipelines.json with stage-based DAGs."""
        config = {
            "dags": {
                "dag_a": [
                    {"script_a1.py": ["dt"]},
                    {"script_a2.py": ["dt", "ts"]},
                ],
                "dag_b": [
                    {"script_b1.py": ["dt"]},
                ],
            }
        }
        (tmp_path / "pipelines.json").write_text(json.dumps(config))
        result = load_pipelines_config(tmp_path)

        assert result is not None
        assert "dag_a" in result
        assert "dag_b" in result
        # dag_a has 2 stages
        assert len(result["dag_a"]) == 2
        # Stage 0: {script_a1: ["dt"]}
        assert result["dag_a"][0] == {tmp_path / "script_a1.py": ["dt"]}
        # Stage 1: {script_a2: ["dt", "ts"]}
        assert result["dag_a"][1] == {tmp_path / "script_a2.py": ["dt", "ts"]}
        # dag_b has 1 stage
        assert result["dag_b"][0] == {tmp_path / "script_b1.py": ["dt"]}

    def test_multi_script_stage(self, tmp_path):
        """Multiple scripts in one stage (parallel execution)."""
        config = {
            "dags": {
                "my_dag": [
                    {"root.py": ["dt"]},
                    {"child_a.py": ["dt", "ts"], "child_b.py": ["dt", "ts"]},
                ],
            }
        }
        (tmp_path / "pipelines.json").write_text(json.dumps(config))
        result = load_pipelines_config(tmp_path)

        # Stage 1 should have 2 scripts
        stage_1 = result["my_dag"][1]
        assert len(stage_1) == 2
        assert stage_1[tmp_path / "child_a.py"] == ["dt", "ts"]
        assert stage_1[tmp_path / "child_b.py"] == ["dt", "ts"]

    def test_returns_none_when_no_json(self, tmp_path):
        """Returns None when no pipelines.json exists in the directory."""
        result = load_pipelines_config(tmp_path)
        assert result is None

    def test_empty_dags(self, tmp_path):
        """Handle JSON with empty dags section."""
        (tmp_path / "pipelines.json").write_text(json.dumps({"dags": {}}))
        result = load_pipelines_config(tmp_path)

        assert result is not None
        assert result == {}

    def test_paths_are_resolved_relative_to_directory(self, tmp_path):
        """Script paths should be relative to the JSON config's directory."""
        config = {"dags": {"my_dag": [{"subdir_script.py": ["dt"]}]}}
        (tmp_path / "pipelines.json").write_text(json.dumps(config))
        result = load_pipelines_config(tmp_path)

        stage_0 = result["my_dag"][0]
        assert tmp_path / "subdir_script.py" in stage_0


class TestSortPipelines:
    """Tests for sort_pipelines with DAG stages and modes."""

    def test_dag_stages_produce_correct_runs(self, tmp_path):
        """DAG stages should produce (script, mode) tuples in stage order."""
        script_a = tmp_path / "a.py"
        script_b = tmp_path / "b.py"

        all_dags = {
            "my_dag": [
                {script_a: ["dt"]},
                {script_b: ["dt", "ts"]},
            ]
        }
        pipelines = [script_a, script_b]

        plan = sort_pipelines(pipelines, all_dags)

        # 3 runs total: a:dt, b:dt, b:ts
        assert len(plan.runs) == 3
        assert plan.runs[0] == (script_a, "dt")
        assert plan.runs[1] == (script_b, "dt")
        assert plan.runs[2] == (script_b, "ts")
        # All in same DAG
        assert plan.group_ids[(script_a, "dt")] == "my_dag"
        assert plan.group_ids[(script_b, "dt")] == "my_dag"
        assert plan.group_ids[(script_b, "ts")] == "my_dag"
        # Stage 0 has no inputs, stage 1 depends on stage 0
        assert plan.deps[(script_a, "dt")] == {"outputs": ["my_dag:stage_0"], "inputs": []}
        assert plan.deps[(script_b, "dt")] == {"outputs": ["my_dag:stage_1"], "inputs": ["my_dag:stage_0"]}
        assert plan.deps[(script_b, "ts")] == {"outputs": ["my_dag:stage_1"], "inputs": ["my_dag:stage_0"]}

    def test_mode_filter_keeps_only_matching(self, tmp_path):
        """Filter mode (dt/ts) should only keep scripts that declare that mode."""
        script_a = tmp_path / "a.py"
        script_b = tmp_path / "b.py"

        all_dags = {
            "my_dag": [
                {script_a: ["dt"]},
                {script_b: ["dt", "ts"]},
            ]
        }
        pipelines = [script_a, script_b]

        plan = sort_pipelines(pipelines, all_dags, mode="dt")

        # Both scripts declare "dt", so 2 runs: a:dt, b:dt (ts filtered out)
        assert len(plan.runs) == 2
        assert plan.runs[0] == (script_a, "dt")
        assert plan.runs[1] == (script_b, "dt")

    def test_mode_filter_skips_non_matching(self, tmp_path):
        """Filter mode should skip scripts that don't declare the requested mode."""
        script_a = tmp_path / "a.py"
        script_b = tmp_path / "b.py"

        all_dags = {
            "my_dag": [
                {script_a: ["dt"]},
                {script_b: ["dt", "ts"]},
            ]
        }
        pipelines = [script_a, script_b]

        plan = sort_pipelines(pipelines, all_dags, mode="ts")

        # Only script_b declares "ts"
        assert len(plan.runs) == 1
        assert plan.runs[0] == (script_b, "ts")

    def test_override_mode_ignores_json_modes(self, tmp_path):
        """Override mode (promote/test_promote) should run every unique script once."""
        script_a = tmp_path / "a.py"
        script_b = tmp_path / "b.py"

        all_dags = {
            "my_dag": [
                {script_a: ["dt"]},
                {script_b: ["dt", "ts"]},
            ]
        }
        pipelines = [script_a, script_b]

        plan = sort_pipelines(pipelines, all_dags, mode="promote")

        # Both scripts run once with promote, regardless of JSON modes
        assert len(plan.runs) == 2
        assert plan.runs[0] == (script_a, "promote")
        assert plan.runs[1] == (script_b, "promote")
        # Override mode has no DAG dependencies
        assert plan.deps[(script_a, "promote")] == {"outputs": [], "inputs": []}
        assert plan.deps[(script_b, "promote")] == {"outputs": [], "inputs": []}

    def test_override_mode_deduplicates_scripts(self, tmp_path):
        """Override mode should deduplicate scripts that appear in multiple stages."""
        script_a = tmp_path / "a.py"
        script_b = tmp_path / "b.py"

        all_dags = {
            "my_dag": [
                {script_a: ["dt"]},
                {script_a: ["ts"], script_b: ["dt", "ts"]},
            ]
        }
        pipelines = [script_a, script_b]

        plan = sort_pipelines(pipelines, all_dags, mode="promote")

        # script_a appears in 2 stages but should only run once
        assert len(plan.runs) == 2
        assert plan.runs[0] == (script_a, "promote")
        assert plan.runs[1] == (script_b, "promote")

    def test_multiple_dags(self, tmp_path):
        """Multiple DAGs should each produce runs independently."""
        script_a = tmp_path / "a.py"
        script_b = tmp_path / "b.py"

        all_dags = {
            "dag_a": [{script_a: ["dt"]}],
            "dag_b": [{script_b: ["dt", "ts"]}],
        }
        pipelines = [script_a, script_b]

        plan = sort_pipelines(pipelines, all_dags)

        assert len(plan.runs) == 3
        assert plan.group_ids[(script_a, "dt")] == "dag_a"
        assert plan.group_ids[(script_b, "dt")] == "dag_b"
        assert plan.group_ids[(script_b, "ts")] == "dag_b"
        assert len(plan.display_lines) == 2
        # Each dag has only stage 0, so no inputs
        assert plan.deps[(script_a, "dt")]["inputs"] == []
        assert plan.deps[(script_b, "dt")]["inputs"] == []

    def test_parallel_stage_display(self, tmp_path):
        """Stage with multiple scripts should show parallel notation."""
        script_a = tmp_path / "xgb.py"
        script_b = tmp_path / "pytorch.py"
        script_c = tmp_path / "chemprop.py"

        all_dags = {
            "my_dag": [
                {script_a: ["dt"]},
                {script_b: ["dt", "ts"], script_c: ["dt"]},
            ]
        }
        pipelines = [script_a, script_b, script_c]

        plan = sort_pipelines(pipelines, all_dags)

        assert len(plan.display_lines) == 1
        # Stage separator is " --> ", within-stage separator is " | "
        assert " --> " in plan.display_lines[0]
        assert "xgb:dt" in plan.display_lines[0]

    def test_filtered_pipelines_only_includes_selected(self, tmp_path):
        """Only pipelines in the selected set should appear."""
        script_a = tmp_path / "a.py"
        script_b = tmp_path / "b.py"
        script_c = tmp_path / "c.py"

        all_dags = {
            "my_dag": [
                {script_a: ["dt"]},
                {script_b: ["dt"], script_c: ["dt"]},
            ]
        }
        # Only select a subset
        pipelines = [script_a, script_b]

        plan = sort_pipelines(pipelines, all_dags)

        assert len(plan.runs) == 2
        assert (script_a, "dt") in plan.runs
        assert (script_b, "dt") in plan.runs
        assert (script_c, "dt") not in plan.runs

    def test_standalone_scripts_included_with_mode(self, tmp_path):
        """Scripts not in any DAG should appear as standalone runs when mode is set."""
        script_a = tmp_path / "a.py"
        script_standalone = tmp_path / "standalone.py"

        all_dags = {"my_dag": [{script_a: ["dt"]}]}
        pipelines = [script_a, script_standalone]

        plan = sort_pipelines(pipelines, all_dags, mode="dt")

        assert len(plan.runs) == 2
        assert plan.runs[0] == (script_a, "dt")
        assert plan.runs[1] == (script_standalone, "dt")
        assert plan.group_ids[(script_standalone, "dt")] is None
        assert plan.deps[(script_standalone, "dt")] == {"outputs": [], "inputs": []}
        assert any("standalone" in line for line in plan.display_lines)

    def test_empty_pipelines(self):
        """Empty pipeline list should return empty results."""
        plan = sort_pipelines([], {})

        assert plan.runs == []
        assert plan.group_ids == {}
        assert plan.display_lines == []
        assert plan.deps == {}

    def test_empty_dags_standalone_with_mode(self, tmp_path):
        """Standalone script with empty dags and mode should produce a run."""
        script = tmp_path / "orphan.py"

        plan = sort_pipelines([script], {}, mode="dt")

        assert len(plan.runs) == 1
        assert plan.runs[0] == (script, "dt")
        assert plan.group_ids[(script, "dt")] is None


class TestExcludedScripts:
    """Tests verifying that scripts not in pipelines.json are excluded."""

    def test_excluded_script_not_discovered(self, tmp_path):
        """Scripts not listed in pipelines.json should not appear in results."""
        config = {"dags": {"my_dag": [{"included.py": ["dt"]}]}}
        (tmp_path / "pipelines.json").write_text(json.dumps(config))
        (tmp_path / "included.py").write_text("# included\n")
        (tmp_path / "excluded.py").write_text("# excluded\n")

        dags = load_pipelines_config(tmp_path)

        all_scripts = set()
        for stages in dags.values():
            for stage in stages:
                all_scripts.update(stage.keys())

        assert tmp_path / "included.py" in all_scripts
        assert tmp_path / "excluded.py" not in all_scripts

    def test_realistic_multi_dag_config(self, tmp_path):
        """Realistic config with multiple DAGs, stages, and scripts parses correctly."""
        config = {
            "dags": {
                "free_class": [
                    {"free_class_xgb_1.py": ["dt"]},
                    {
                        "free_class_pytorch_1.py": ["dt", "ts"],
                        "free_class_chemprop_1.py": ["dt", "ts"],
                    },
                ],
                "free_reg": [
                    {"free_reg_xgb_1.py": ["dt"]},
                    {
                        "free_reg_pytorch_1.py": ["dt", "ts"],
                        "free_reg_chemprop_1.py": ["dt", "ts"],
                    },
                ],
            }
        }
        (tmp_path / "pipelines.json").write_text(json.dumps(config))

        dags = load_pipelines_config(tmp_path)

        all_scripts = set()
        for stages in dags.values():
            for stage in stages:
                all_scripts.update(stage.keys())

        assert len(dags) == 2
        assert len(all_scripts) == 6
        script_names = [s.name for s in all_scripts]
        assert "free_class_xgb_1.py" in script_names
        assert "free_reg_pytorch_1.py" in script_names


class TestGetAllPipelines:
    """Tests for get_all_pipelines with nested directory structures."""

    def test_nested_json_discovered(self, tmp_path):
        """pipelines.json in a deeply nested directory should be found."""
        leaf = tmp_path / "top" / "middle" / "leaf"
        leaf.mkdir(parents=True)
        (leaf / "script_a.py").write_text("# a\n")
        (leaf / "script_b.py").write_text("# b\n")
        (leaf / "standalone.py").write_text("# standalone script\n")
        config = {"dags": {"my_dag": [{"script_a.py": ["dt"]}, {"script_b.py": ["dt"]}]}}
        (leaf / "pipelines.json").write_text(json.dumps(config))

        with patch("os.getcwd", return_value=str(tmp_path)):
            os.chdir(tmp_path)
            pipelines, all_dags = get_all_pipelines()

        script_names = [p.name for p in pipelines]
        assert "script_a.py" in script_names
        assert "script_b.py" in script_names
        # Standalone scripts are also discovered
        assert "standalone.py" in script_names
        assert "my_dag" in all_dags

    def test_standalone_scripts_discovered(self, tmp_path):
        """Scripts without pipelines.json should be discovered as standalone."""
        json_dir = tmp_path / "project" / "assay_a"
        json_dir.mkdir(parents=True)
        (json_dir / "included.py").write_text("# in JSON\n")
        config = {"dags": {"dag_a": [{"included.py": ["dt"]}]}}
        (json_dir / "pipelines.json").write_text(json.dumps(config))

        standalone_dir = tmp_path / "project" / "assay_b"
        standalone_dir.mkdir(parents=True)
        (standalone_dir / "standalone_script.py").write_text("# standalone\n")

        os.chdir(tmp_path)
        pipelines, all_dags = get_all_pipelines()

        script_names = [p.name for p in pipelines]
        assert "included.py" in script_names
        assert "standalone_script.py" in script_names
        assert "dag_a" in all_dags

    def test_dunder_files_excluded(self, tmp_path):
        """__init__.py and other dunder files should be excluded."""
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / "real_script.py").write_text("# real\n")

        os.chdir(tmp_path)
        pipelines, all_dags = get_all_pipelines()

        script_names = [p.name for p in pipelines]
        assert "real_script.py" in script_names
        assert "__init__.py" not in script_names


class TestParseScriptName:
    """Tests for parse_script_name filename parsing with optional version."""

    def test_standard_version(self, tmp_path):
        """Standard _1.py suffix parses correctly."""
        basename, version = parse_script_name(tmp_path / "my_script_1.py")
        assert basename == "my_script"
        assert version == "1"

    def test_v_prefix_version(self, tmp_path):
        """_v2.py suffix parses correctly."""
        basename, version = parse_script_name(tmp_path / "my_script_v2.py")
        assert basename == "my_script"
        assert version == "2"

    def test_multi_digit_version(self, tmp_path):
        """Multi-digit version number parses correctly."""
        basename, version = parse_script_name(tmp_path / "my_script_12.py")
        assert basename == "my_script"
        assert version == "12"

    def test_real_pipeline_filename(self, tmp_path):
        """Real pipeline filename parses correctly."""
        basename, version = parse_script_name(tmp_path / "ppb_human_free_reg_xgb_1.py")
        assert basename == "ppb_human_free_reg_xgb"
        assert version == "1"

    def test_no_version_returns_none(self, tmp_path):
        """Filename without version suffix returns None for version."""
        basename, version = parse_script_name(tmp_path / "caco2_er_reg_open_admet.py")
        assert basename == "caco2_er_reg_open_admet"
        assert version is None

    def test_non_numeric_suffix_returns_none(self, tmp_path):
        """Filename ending with non-numeric suffix returns None for version."""
        basename, version = parse_script_name(tmp_path / "my_script_abc.py")
        assert basename == "my_script_abc"
        assert version is None
