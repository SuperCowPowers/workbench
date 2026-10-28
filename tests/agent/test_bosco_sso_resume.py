"""An expired SSO token mid-turn: Bosco renews, answers any tool call left pending, and finishes the turn.

The API rejects a history whose last tool_use has no tool_result, so the retried
turn must see that pairing closed before it calls the model again.
"""

from types import SimpleNamespace

from botocore.exceptions import TokenRetrievalError

from workbench.agent import bosco as bosco_mod
from workbench.utils import sso_utils


def test_turn_resumes_after_renewal(monkeypatch):
    history = []
    monkeypatch.setattr(bosco_mod, "_history", history)
    monkeypatch.setattr(bosco_mod, "job_updates", lambda prompt: prompt)
    monkeypatch.setattr(bosco_mod, "_namespace", lambda: {})
    monkeypatch.setattr(sso_utils, "sso_renew", lambda profile=None: True)

    seen = []

    def run_turn(namespace):
        seen.append(list(history))
        if len(seen) == 1:
            history.append({"role": "assistant", "content": [SimpleNamespace(type="tool_use", id="toolu_1")]})
            raise TokenRetrievalError(provider="sso", error_msg="Token has expired and refresh failed")
        history.append({"role": "assistant", "content": []})

    monkeypatch.setattr(bosco_mod, "_run_turn", run_turn)
    bosco_mod._ask("what models do we have")

    assert len(seen) == 2
    resumed = seen[1][-1]
    assert resumed["role"] == "user"
    assert [r["tool_use_id"] for r in resumed["content"]] == ["toolu_1"]


def test_declined_renewal_reports_and_closes(monkeypatch, capsys):
    history = []
    monkeypatch.setattr(bosco_mod, "_history", history)
    monkeypatch.setattr(bosco_mod, "job_updates", lambda prompt: prompt)
    monkeypatch.setattr(bosco_mod, "_namespace", lambda: {})
    monkeypatch.setattr(sso_utils, "sso_renew", lambda profile=None: False)

    def run_turn(namespace):
        history.append({"role": "assistant", "content": [SimpleNamespace(type="tool_use", id="toolu_1")]})
        raise TokenRetrievalError(provider="sso", error_msg="Token has expired and refresh failed")

    monkeypatch.setattr(bosco_mod, "_run_turn", run_turn)
    bosco_mod._ask("what models do we have")

    assert "TokenRetrievalError" in capsys.readouterr().out
    assert history[-1]["role"] == "user"
