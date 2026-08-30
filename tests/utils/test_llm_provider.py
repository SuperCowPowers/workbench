"""Tests for provider selection in workbench.utils.llm_utils.

The invariant these guard: an AWS-connected session always reaches Claude through
Bedrock, so Bosco's traffic stays inside the user's own account. No key in the
environment or the config can divert it. The key-based lanes serve local mode
only, where there is no account for the traffic to stay inside.
"""

import pytest

from workbench.utils import llm_utils
from workbench.utils.config_manager import ConfigManager


@pytest.fixture(autouse=True)
def clear_provider_cache():
    """llm_provider() is cached for the session, so each test needs a clean slate."""
    llm_utils.llm_provider.cache_clear()
    yield
    llm_utils.llm_provider.cache_clear()


def configure(monkeypatch, aws: bool, anthropic_key: str = None, llm_key: str = None, llm_url: str = None):
    """Point llm_provider() at a synthetic world rather than the real config.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        aws (bool): Whether a usable AWS site config is present.
        anthropic_key (str, optional): Value for the ANTHROPIC_API_KEY env var.
        llm_key (str, optional): The SuperCowPowers trial key.
        llm_url (str, optional): The SCP proxy endpoint.
    """
    config = {llm_utils.LLM_KEY: llm_key, llm_utils.LLM_URL: llm_url}
    monkeypatch.setattr(ConfigManager, "config_okay", lambda self: aws)
    monkeypatch.setattr(ConfigManager, "get_config", lambda self, key, default=None: config.get(key, default))
    if anthropic_key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
    else:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm_utils.llm_provider.cache_clear()


#
# AWS mode: Bedrock, whatever else is lying around
#


def test_aws_mode_uses_bedrock(monkeypatch):
    """A connected account reaches Claude through Bedrock"""
    configure(monkeypatch, aws=True)
    assert llm_utils.llm_provider() == "bedrock"


def test_aws_mode_ignores_anthropic_key(monkeypatch):
    """An ANTHROPIC_API_KEY left in the environment does not divert an AWS session"""
    configure(monkeypatch, aws=True, anthropic_key="sk-ant-test")
    assert llm_utils.llm_provider() == "bedrock"


def test_aws_mode_ignores_trial_key(monkeypatch):
    """A trial key does not divert an AWS session through the SCP proxy"""
    configure(monkeypatch, aws=True, llm_key="wb-test", llm_url="https://proxy.invalid/v1")
    assert llm_utils.llm_provider() == "bedrock"


def test_aws_mode_ignores_every_key(monkeypatch):
    """Both keys present at once still leaves an AWS session on Bedrock"""
    configure(monkeypatch, aws=True, anthropic_key="sk-ant-test", llm_key="wb-test", llm_url="https://p.invalid/v1")
    assert llm_utils.llm_provider() == "bedrock"


#
# Local mode: the key lanes
#


def test_local_mode_no_keys(monkeypatch):
    """Local mode with nothing configured has no path to Claude"""
    configure(monkeypatch, aws=False)
    assert llm_utils.llm_provider() == "none"


def test_local_mode_anthropic_key(monkeypatch):
    """Local mode uses the user's own key when they have one"""
    configure(monkeypatch, aws=False, anthropic_key="sk-ant-test")
    assert llm_utils.llm_provider() == "anthropic"


def test_local_mode_trial_key(monkeypatch):
    """Local mode falls back to the SCP trial lane"""
    configure(monkeypatch, aws=False, llm_key="wb-test", llm_url="https://proxy.invalid/v1")
    assert llm_utils.llm_provider() == "trial"


def test_local_mode_trial_key_without_url(monkeypatch):
    """A trial key alone is not a path -- the proxy endpoint has to be named"""
    configure(monkeypatch, aws=False, llm_key="wb-test")
    assert llm_utils.llm_provider() == "none"


def test_local_mode_prefers_own_key_over_trial(monkeypatch):
    """A user's own key wins over the subsidized lane"""
    configure(monkeypatch, aws=False, anthropic_key="sk-ant-test", llm_key="wb-test", llm_url="https://p.invalid/v1")
    assert llm_utils.llm_provider() == "anthropic"


#
# Model ids and availability
#


def test_default_model_is_an_inference_profile_on_bedrock(monkeypatch):
    """Bedrock takes a prefixed inference profile id"""
    configure(monkeypatch, aws=True)
    assert llm_utils.default_model() == f"{llm_utils.BEDROCK_PREFIX}{llm_utils.CLAUDE_MODELS[0]}"


def test_default_model_is_bare_off_bedrock(monkeypatch):
    """The direct API takes the bare model name"""
    configure(monkeypatch, aws=False, anthropic_key="sk-ant-test")
    assert llm_utils.default_model() == llm_utils.CLAUDE_MODELS[0]


def test_default_model_accepts_a_pinned_provider(monkeypatch):
    """A caller can ask for another provider's id form (the Bedrock verifier does)"""
    configure(monkeypatch, aws=False, anthropic_key="sk-ant-test")
    assert llm_utils.default_model("bedrock").startswith(llm_utils.BEDROCK_PREFIX)


def test_no_provider_is_unavailable(monkeypatch):
    """Nothing configured means Bosco stays dark"""
    configure(monkeypatch, aws=False)
    assert llm_utils.llm_available() is False


def test_key_providers_are_available_without_a_round_trip(monkeypatch):
    """A key is taken at its word, so REPL startup pays no network cost.

    An obviously bogus key still reports available -- a real one is only proven on
    the first call, which is the trade that keeps startup fast.
    """
    configure(monkeypatch, aws=False, anthropic_key="sk-ant-not-a-real-key")
    assert llm_utils.llm_available() is True


def test_client_without_a_provider_raises(monkeypatch):
    """Building a client with no path configured fails loudly"""
    configure(monkeypatch, aws=False)
    with pytest.raises(RuntimeError, match="No path to Claude"):
        llm_utils.claude_client()
