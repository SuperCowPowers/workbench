"""Bosco's model and effort come from the user's Workbench config, and a saved change applies at once."""

import json
from types import SimpleNamespace

import pytest

from workbench.agent import bosco as bosco_mod
from workbench.utils.config_manager import ConfigManager

SONNET = "claude-sonnet-5"
USAGE = SimpleNamespace(input_tokens=1000, output_tokens=100)


@pytest.fixture
def config(monkeypatch):
    """The ConfigManager singleton holding an empty config."""
    settings = {}
    monkeypatch.setattr(ConfigManager(), "config", settings)
    return settings


@pytest.fixture
def fresh_usage(monkeypatch):
    """Zeroed session totals, restored afterwards."""
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "calls": 0, "cost_usd": 0.0}
    monkeypatch.setattr(bosco_mod.bosco, "usage", usage)
    return usage


def test_defaults_with_nothing_configured(config):
    assert bosco_mod._model() == bosco_mod.CLAUDE_MODELS[0]
    assert bosco_mod._effort() == bosco_mod.DEFAULT_EFFORT


def test_config_sets_model_and_effort(config):
    config.update({"BOSCO_MODEL": SONNET, "BOSCO_EFFORT": "low"})
    assert bosco_mod._model() == SONNET
    assert bosco_mod._effort() == "low"


@pytest.mark.parametrize("model", ["claude-opus-99", "", None])
def test_unknown_model_falls_back(config, model):
    config["BOSCO_MODEL"] = model
    assert bosco_mod._model() == bosco_mod.CLAUDE_MODELS[0]


@pytest.mark.parametrize("effort", ["turbo", "", None])
def test_unknown_effort_falls_back(config, effort):
    config["BOSCO_EFFORT"] = effort
    assert bosco_mod._effort() == bosco_mod.DEFAULT_EFFORT


def test_every_effort_level_is_accepted(config):
    for level in bosco_mod.EFFORT_LEVELS:
        config["BOSCO_EFFORT"] = level
        assert bosco_mod._effort() == level


def test_default_model_has_a_rate(config):
    """An unpriced default would raise a KeyError in cost tracking on every turn."""
    assert bosco_mod.CLAUDE_MODELS[0] in bosco_mod._RATES


def test_usage_is_priced_at_the_model_that_ran(fresh_usage):
    bosco_mod._track_usage(USAGE, SONNET)
    rates = bosco_mod._RATES[SONNET]
    assert fresh_usage["cost_usd"] == round(1000 * rates["input"] + 100 * rates["output"], 4)


def test_turn_sends_the_configured_settings(config, fresh_usage, monkeypatch):
    config.update({"BOSCO_MODEL": SONNET, "BOSCO_EFFORT": "low"})
    sent = {}

    def stream(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(usage=USAGE, content=[], stop_reason="end_turn")

    monkeypatch.setattr(bosco_mod, "message_stream", stream)
    monkeypatch.setattr(bosco_mod, "model_id", lambda name: f"profile.{name}")
    monkeypatch.setattr(bosco_mod, "_history", [{"role": "user", "content": "hi"}])
    monkeypatch.setattr(bosco_mod, "_system_prompt", lambda: "system")

    bosco_mod._run_turn({})
    assert sent["model"] == f"profile.{SONNET}"
    assert sent["output_config"] == {"effort": "low"}


def test_saved_change_applies_now_and_reaches_the_file(tmp_path, config, monkeypatch):
    site = tmp_path / "site.json"
    site.write_text('{"WORKBENCH_BUCKET": "bucket"}', encoding="utf-8")
    cm = ConfigManager()
    monkeypatch.setattr(cm, "site_config_path", str(site))
    monkeypatch.setattr(cm, "using_default_config", False)

    cm.save_config("BOSCO_MODEL", SONNET)
    cm.save_config("BOSCO_EFFORT", "low")

    assert bosco_mod._model() == SONNET
    assert bosco_mod._effort() == "low"
    assert json.loads(site.read_text(encoding="utf-8")) == {
        "WORKBENCH_BUCKET": "bucket",
        "BOSCO_MODEL": SONNET,
        "BOSCO_EFFORT": "low",
    }


def test_status_shows_both_settings(config, capsys):
    config.update({"BOSCO_MODEL": SONNET, "BOSCO_EFFORT": "low"})
    bosco_mod.bosco()
    output = capsys.readouterr().out
    assert f"model={SONNET}" in output
    assert "effort=low" in output
