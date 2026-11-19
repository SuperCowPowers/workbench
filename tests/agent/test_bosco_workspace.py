"""Bosco's guides and report directories come from the user's Workbench config."""

import pytest

from workbench.agent import tools
from workbench.utils.config_manager import ConfigManager


@pytest.fixture
def config(monkeypatch):
    """The ConfigManager singleton holding an empty, saveable config."""
    settings = {}
    monkeypatch.setattr(ConfigManager(), "config", settings)
    monkeypatch.setattr(ConfigManager(), "using_default_config", False)
    return settings


@pytest.fixture
def guides_dir(tmp_path, config):
    """A configured guides directory holding one user guide."""
    path = tmp_path / "guides"
    path.mkdir()
    (path / "team_conventions.md").write_text("# Team Conventions\n\n> how our team names things\n\nBody.\n")
    config[tools.GUIDES_DIR_KEY] = str(path)
    return path


def test_unset_directories_are_asked_for(config):
    text = tools.workspace_text()
    assert f'save_config("{tools.GUIDES_DIR_KEY}"' in text and "~/<local dir>/agent/guides" in text
    assert f'save_config("{tools.REPORT_DIR_KEY}"' in text and "<path to your repo>/reports" in text


def test_guides_ask_up_front_and_reports_on_first_report(config):
    guides_ask, report_ask = tools.workspace_text().split("- **Reports**")
    assert "Before your first piece of work" in guides_ask
    assert "first time you are about to write a report" in report_ask


def test_only_the_unset_directory_is_asked_for(config, tmp_path):
    config[tools.REPORT_DIR_KEY] = str(tmp_path)
    text = tools.workspace_text()
    assert f"`{tmp_path}`" in text
    assert f'save_config("{tools.GUIDES_DIR_KEY}"' in text
    assert f'save_config("{tools.REPORT_DIR_KEY}"' not in text


def test_declined_directories_are_not_asked_for(config):
    config.update({tools.GUIDES_DIR_KEY: "", tools.REPORT_DIR_KEY: ""})
    assert tools.workspace_text() == ""


def test_session_only_config_never_asks(config, monkeypatch):
    monkeypatch.setattr(ConfigManager(), "using_default_config", True)
    assert tools.workspace_text() == ""


def test_user_guides_join_the_index(guides_dir):
    index = tools.guide_index()
    assert f"user guides ({guides_dir})/" in index
    assert "team_conventions" in index and "how our team names things" in index


def test_read_guide_finds_a_user_guide(guides_dir):
    assert "Body." in tools.read_guide("team_conventions")


def test_user_guide_wins_over_a_bundled_one(guides_dir):
    (guides_dir / "plotting.md").write_text("# Plotting\n\n> ours\n\nTeam plotting rules.\n")
    assert "Team plotting rules." in tools.read_guide("plotting")


def test_missing_guides_directory_adds_nothing(config, tmp_path):
    config[tools.GUIDES_DIR_KEY] = str(tmp_path / "not_there")
    assert tools.user_guide_paths() == []
    assert "user guides" not in tools.guide_index()
