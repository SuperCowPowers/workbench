"""The implicit last session: every Bosco turn rewrites a bounded record, and the next
REPL reads back the one the previous REPL left.

The invariants: the record stays under the store's size budget however long the REPL
runs, a failed write never reaches the user, and once this REPL starts rewriting the
record, the previous session is still what Bosco reads.
"""

import json
from types import SimpleNamespace

from botocore.exceptions import TokenRetrievalError

from workbench.agent import bosco as bosco_mod
from workbench.agent.tools import dispatch
from workbench.utils import bosco_utils


def text(value: str):
    return SimpleNamespace(type="text", text=value)


def answer_with(monkeypatch, *replies: str):
    """Make each turn append one assistant message with the given text, in order."""
    queue = list(replies)

    def run_turn(namespace):
        bosco_mod._history.append({"role": "assistant", "content": [text(queue.pop(0))]})

    monkeypatch.setattr(bosco_mod, "_history", [])
    monkeypatch.setattr(bosco_mod, "job_updates", lambda prompt: prompt)
    monkeypatch.setattr(bosco_mod, "_namespace", lambda: {})
    monkeypatch.setattr(bosco_mod, "_run_turn", run_turn)


def stored(store) -> dict:
    return store.get(bosco_utils.last_session_path(), warn=False)


def test_each_turn_saves(monkeypatch, isolated_session_store):
    answer_with(monkeypatch, "You have 3 models.", "aqsol-v2 has the best MAE.")
    bosco_mod._ask("what models do we have")
    bosco_mod._ask("which is best")

    record = stored(isolated_session_store)
    assert record["exchanges"] == [
        {"prompt": "what models do we have", "reply": "You have 3 models."},
        {"prompt": "which is best", "reply": "aqsol-v2 has the best MAE."},
    ]


def test_tool_calls_are_left_out(monkeypatch, isolated_session_store):
    answer_with(monkeypatch)

    def run_turn(namespace):
        tool = SimpleNamespace(type="tool_use", id="toolu_1")
        bosco_mod._history.append({"role": "assistant", "content": [text("Checking."), tool]})
        bosco_mod._history.append({"role": "user", "content": [{"type": "tool_result", "content": "big output"}]})
        bosco_mod._history.append({"role": "assistant", "content": [text("Found 3.")]})

    monkeypatch.setattr(bosco_mod, "_run_turn", run_turn)
    bosco_mod._ask("count the models")
    assert stored(isolated_session_store)["exchanges"][0]["reply"] == "Checking.\n\nFound 3."


def test_record_stays_bounded(monkeypatch, isolated_session_store):
    turns = 100
    answer_with(monkeypatch, *(["x" * 2000] * turns))
    for i in range(turns):
        bosco_mod._ask(f"question {i} " + "y" * 2000)

    record = stored(isolated_session_store)
    assert len(json.dumps(record)) <= bosco_utils.LAST_SESSION_CHARS
    assert record["exchanges"][-1]["prompt"].startswith("question 99 ")
    assert len(bosco_mod._exchanges) == len(record["exchanges"])  # memory is bounded too


def test_failed_turn_is_still_recorded(monkeypatch, isolated_session_store, capsys):
    answer_with(monkeypatch)

    def run_turn(namespace):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(bosco_mod, "_run_turn", run_turn)
    bosco_mod._ask("deploy it")
    assert stored(isolated_session_store)["exchanges"] == [{"prompt": "deploy it", "reply": ""}]


def test_failed_write_stays_quiet(monkeypatch, isolated_session_store, capsys):
    def expired(name, value):
        raise TokenRetrievalError(provider="sso", error_msg="Token has expired and refresh failed")

    monkeypatch.setattr(isolated_session_store, "upsert", expired)
    answer_with(monkeypatch, "You have 3 models.")
    bosco_mod._ask("what models do we have")
    assert "TokenRetrievalError" not in capsys.readouterr().out


def test_previous_session_survives_this_repls_saves(monkeypatch, isolated_session_store):
    earlier = [bosco_utils.exchange("train a logd model", "Started logd-xgb-v1.")]
    bosco_utils.save_last_session(earlier)

    answer_with(monkeypatch, "Hello again.")
    bosco_mod._ask("hi")

    assert stored(isolated_session_store)["exchanges"][-1]["prompt"] == "hi"
    previous = dispatch("read_last_session", {}, {})
    assert "train a logd model" in previous and "Started logd-xgb-v1." in previous
    assert "hi" not in previous.split("\n")


def test_hint_only_with_a_previous_session(isolated_session_store):
    assert bosco_mod._previous_hint() == ""
    assert dispatch("read_last_session", {}, {}) == "No previous session recorded."

    bosco_utils.save_last_session([bosco_utils.exchange("train a logd model", "Started.")])
    bosco_utils.previous_session.cache_clear()
    bosco_mod._previous_hint.cache_clear()
    assert "read_last_session" in bosco_mod._previous_hint()
