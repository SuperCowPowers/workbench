"""Tests for ml_pipeline_launcher.py - pipelines.yaml loading and sort_pipelines."""

from pathlib import Path

import os
from unittest.mock import patch

from workbench.scripts.ml_pipeline_launcher import (
    get_all_pipelines,
    load_pipelines_yaml,
    sort_pipelines,
)


class TestLoadPipelinesYaml:
    """Tests for loading pipelines.yaml from a directory."""

    def test_load_valid_yaml(self, tmp_path):
        """Load a valid pipelines.yaml with two chains."""
        yaml_content = """
chains:
  chain_a:
    - script_a1.py
    - script_a2.py
  chain_b:
    - script_b1.py
"""
        (tmp_path / "pipelines.yaml").write_text(yaml_content)
        result = load_pipelines_yaml(tmp_path)

        assert result is not None
        assert "chain_a" in result
        assert "chain_b" in result
        assert result["chain_a"] == [tmp_path / "script_a1.py", tmp_path / "script_a2.py"]
        assert result["chain_b"] == [tmp_path / "script_b1.py"]

    def test_returns_none_when_no_yaml(self, tmp_path):
        """Returns None when no pipelines.yaml exists in the directory."""
        result = load_pipelines_yaml(tmp_path)
        assert result is None

    def test_empty_chains(self, tmp_path):
        """Handle yaml with empty chains section."""
        (tmp_path / "pipelines.yaml").write_text("chains: {}\n")
        result = load_pipelines_yaml(tmp_path)

        assert result is not None
        assert result == {}

    def test_paths_are_resolved_relative_to_directory(self, tmp_path):
        """Script paths should be relative to the yaml's directory."""
        yaml_content = """
chains:
  my_chain:
    - subdir_script.py
"""
        (tmp_path / "pipelines.yaml").write_text(yaml_content)
        result = load_pipelines_yaml(tmp_path)

        assert result["my_chain"] == [tmp_path / "subdir_script.py"]


class TestSortPipelines:
    """Tests for sort_pipelines with yaml chains."""

    def test_yaml_chains_ordered_correctly(self, tmp_path):
        """Pipelines from yaml chains should be in chain order."""
        script_a = tmp_path / "a.py"
        script_b = tmp_path / "b.py"
        script_c = tmp_path / "c.py"

        all_chains = {"my_chain": [script_a, script_b, script_c]}
        pipelines = [script_c, script_a, script_b]  # Deliberately out of order

        sorted_pipelines, group_id_map, chain_lines = sort_pipelines(pipelines, all_chains)

        assert sorted_pipelines == [script_a, script_b, script_c]
        assert group_id_map[script_a] == "my_chain"
        assert group_id_map[script_b] == "my_chain"
        assert group_id_map[script_c] == "my_chain"

    def test_multiple_chains(self, tmp_path):
        """Multiple chains should each be sorted independently."""
        script_a1 = tmp_path / "a1.py"
        script_a2 = tmp_path / "a2.py"
        script_b1 = tmp_path / "b1.py"
        script_b2 = tmp_path / "b2.py"

        all_chains = {
            "chain_a": [script_a1, script_a2],
            "chain_b": [script_b1, script_b2],
        }
        pipelines = [script_b2, script_a2, script_b1, script_a1]

        sorted_pipelines, group_id_map, chain_lines = sort_pipelines(pipelines, all_chains)

        # chain_a should come first (dict order), then chain_b
        assert sorted_pipelines == [script_a1, script_a2, script_b1, script_b2]
        assert group_id_map[script_a1] == "chain_a"
        assert group_id_map[script_b1] == "chain_b"
        assert len(chain_lines) == 2

    def test_chain_lines_format(self, tmp_path):
        """Chain display lines should use ' --> ' separator."""
        script_a = tmp_path / "xgb_model.py"
        script_b = tmp_path / "pytorch_model.py"

        all_chains = {"my_chain": [script_a, script_b]}
        pipelines = [script_a, script_b]

        _, _, chain_lines = sort_pipelines(pipelines, all_chains)

        assert len(chain_lines) == 1
        assert "xgb_model --> pytorch_model" in chain_lines[0]

    def test_filtered_pipelines_only_includes_selected(self, tmp_path):
        """Only pipelines in the selected set should appear, even if chain has more."""
        script_a = tmp_path / "a.py"
        script_b = tmp_path / "b.py"
        script_c = tmp_path / "c.py"

        all_chains = {"my_chain": [script_a, script_b, script_c]}
        # Only select a subset
        pipelines = [script_a, script_b]

        sorted_pipelines, group_id_map, chain_lines = sort_pipelines(pipelines, all_chains)

        assert sorted_pipelines == [script_a, script_b]
        assert script_c not in group_id_map

    def test_no_chains_falls_back_to_workbench_batch(self, tmp_path):
        """When no yaml chains, should fall back to WORKBENCH_BATCH parsing."""
        # Create a script with WORKBENCH_BATCH
        script = tmp_path / "test_script.py"
        script.write_text('WORKBENCH_BATCH = {"outputs": ["test-output"]}\n')

        sorted_pipelines, group_id_map, chain_lines = sort_pipelines([script], {})

        assert sorted_pipelines == [script]
        # group_id should come from WORKBENCH_BATCH parsing
        assert group_id_map[script] == "test-output"

    def test_empty_pipelines(self):
        """Empty pipeline list should return empty results."""
        sorted_pipelines, group_id_map, chain_lines = sort_pipelines([], {})

        assert sorted_pipelines == []
        assert group_id_map == {}
        assert chain_lines == []

    def test_mixed_yaml_and_workbench_batch(self, tmp_path):
        """Pipelines from yaml chains and WORKBENCH_BATCH should both work."""
        # YAML chain scripts
        yaml_script = tmp_path / "yaml_script.py"
        yaml_script.write_text("# yaml managed\n")

        # WORKBENCH_BATCH script
        wb_script = tmp_path / "wb_script.py"
        wb_script.write_text('WORKBENCH_BATCH = {"outputs": ["wb-output"]}\n')

        all_chains = {"yaml_chain": [yaml_script]}
        pipelines = [yaml_script, wb_script]

        sorted_pipelines, group_id_map, chain_lines = sort_pipelines(pipelines, all_chains)

        assert yaml_script in sorted_pipelines
        assert wb_script in sorted_pipelines
        assert group_id_map[yaml_script] == "yaml_chain"
        assert group_id_map[wb_script] == "wb-output"


class TestExcludedScripts:
    """Tests verifying that scripts not in pipelines.yaml are excluded."""

    def test_excluded_script_not_discovered(self, tmp_path):
        """Scripts not listed in pipelines.yaml should not appear in results."""
        yaml_content = """
chains:
  my_chain:
    - included.py
"""
        (tmp_path / "pipelines.yaml").write_text(yaml_content)
        # Create both scripts on disk
        (tmp_path / "included.py").write_text("# included\n")
        (tmp_path / "excluded.py").write_text("# excluded\n")

        chains = load_pipelines_yaml(tmp_path)

        # Only included.py should be in the chains
        all_scripts = []
        for chain_scripts in chains.values():
            all_scripts.extend(chain_scripts)

        assert tmp_path / "included.py" in all_scripts
        assert tmp_path / "excluded.py" not in all_scripts

    def test_ppb_human_all_scripts_in_yaml(self):
        """Verify the actual ppb_human pipelines.yaml includes all 7 scripts."""
        ppb_human_dir = Path("/Users/briford/work/ideaya/promoted_ml_pipelines/ml_pipelines/Binding/ppb_human")
        if not (ppb_human_dir / "pipelines.yaml").exists():
            return  # Skip if not on this machine

        chains = load_pipelines_yaml(ppb_human_dir)
        all_scripts = []
        for chain_scripts in chains.values():
            all_scripts.extend(chain_scripts)

        script_names = [s.name for s in all_scripts]
        assert "ppb_human_free_reg_xgb_1.py" in script_names
        assert "ppb_human_free_reg_pytorch_1.py" in script_names
        assert "ppb_human_free_reg_chemprop_1.py" in script_names
        assert "ppb_human_free_reg_chemeleon_1.py" in script_names
        assert "ppb_human_free_class_xgb_1.py" in script_names
        assert "ppb_human_free_class_pytorch_1.py" in script_names
        assert "ppb_human_free_class_chemprop_1.py" in script_names
        assert len(all_scripts) == 7


class TestGetAllPipelines:
    """Tests for get_all_pipelines with nested directory structures."""

    def test_nested_yaml_discovered(self, tmp_path):
        """pipelines.yaml in a deeply nested directory should be found."""
        # Simulate: cwd/top/middle/leaf/pipelines.yaml
        leaf = tmp_path / "top" / "middle" / "leaf"
        leaf.mkdir(parents=True)
        (leaf / "script_a.py").write_text("# a\n")
        (leaf / "script_b.py").write_text("# b\n")
        (leaf / "excluded.py").write_text("# should not appear\n")
        (leaf / "pipelines.yaml").write_text(
            "chains:\n  my_chain:\n    - script_a.py\n    - script_b.py\n"
        )

        with patch("os.getcwd", return_value=str(tmp_path)):
            os.chdir(tmp_path)
            pipelines, all_chains = get_all_pipelines()

        script_names = [p.name for p in pipelines]
        assert "script_a.py" in script_names
        assert "script_b.py" in script_names
        assert "excluded.py" not in script_names
        assert "my_chain" in all_chains

    def test_mixed_nested_yaml_and_non_yaml_dirs(self, tmp_path):
        """Directories with yaml exclude unlisted scripts; dirs without yaml include all .py."""
        # Dir with yaml (nested)
        yaml_dir = tmp_path / "project" / "assay_a"
        yaml_dir.mkdir(parents=True)
        (yaml_dir / "included.py").write_text("# in yaml\n")
        (yaml_dir / "excluded.py").write_text("# not in yaml\n")
        (yaml_dir / "pipelines.yaml").write_text(
            "chains:\n  chain_a:\n    - included.py\n"
        )

        # Dir without yaml (also nested under same top-level)
        no_yaml_dir = tmp_path / "project" / "assay_b"
        no_yaml_dir.mkdir(parents=True)
        (no_yaml_dir / "legacy_script.py").write_text(
            'WORKBENCH_BATCH = {"outputs": ["legacy-out"]}\n'
        )

        os.chdir(tmp_path)
        pipelines, all_chains = get_all_pipelines()

        script_names = [p.name for p in pipelines]
        assert "included.py" in script_names
        assert "excluded.py" not in script_names
        assert "legacy_script.py" in script_names
        assert "chain_a" in all_chains
