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
        {"prompt": "what models do we have", "reply": "You have 3 models.", "worked": False},
        {"prompt": "which is best", "reply": "aqsol-v2 has the best MAE.", "worked": False},
    ]


def test_tool_calls_are_left_out(monkeypatch, isolated_session_store):
    answer_with(monkeypatch)

    def run_turn(namespace):
        tool = SimpleNamespace(type="tool_use", id="toolu_1", name="run_python")
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
    assert stored(isolated_session_store)["exchanges"] == [{"prompt": "deploy it", "reply": "", "worked": False}]


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


def test_no_previous_session(isolated_session_store):
    assert dispatch("read_last_session", {}, {}) == "No previous session recorded."


def test_startup_teaser(isolated_session_store):
    from workbench.repl.workbench_shell import BOSCO_INVITE, WorkbenchShell

    assert bosco_utils.last_session_teaser() is None
    assert WorkbenchShell.bosco_greeting() == BOSCO_INVITE

    bosco_utils.save_last_session(
        [
            bosco_utils.exchange("train a logd model", "Started."),
            bosco_utils.exchange("compare aqsol-v1 and aqsol-v2 on the analog set, scaffold split", "v2 wins."),
        ]
    )
    bosco_utils.previous_session.cache_clear()
    teaser = bosco_utils.last_session_teaser()
    assert teaser.startswith('Last time (just now): "compare aqsol-v1 and aqsol-v2 on the analog set, scaffold..."')
    assert teaser.endswith('type "last session" to pick up')
    assert teaser in WorkbenchShell.bosco_greeting()


def recall_turn(monkeypatch, reply: str):
    """A turn where Bosco reads the previous session and answers from it."""

    def run_turn(namespace):
        tool = SimpleNamespace(type="tool_use", id="toolu_r", name="read_last_session")
        bosco_mod._history.append({"role": "assistant", "content": [tool]})
        bosco_mod._history.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_r"}]})
        bosco_mod._history.append({"role": "assistant", "content": [text(reply)]})

    monkeypatch.setattr(bosco_mod, "_run_turn", run_turn)


def test_teaser_names_the_last_real_request(isolated_session_store):
    bosco_utils.save_last_session(
        [
            bosco_utils.exchange("retrain logd with scaffold split", "Done.", worked=True),
            bosco_utils.exchange("lets continue", "We were retraining logd."),
            bosco_utils.exchange("thanks", "Anytime!"),
        ]
    )
    assert '"retrain logd with scaffold split"' in bosco_utils.last_session_teaser()


def test_recall_carries_the_thread(monkeypatch, isolated_session_store):
    bosco_utils.save_last_session([bosco_utils.exchange("retrain logd with scaffold split", "Done.", worked=True)])

    answer_with(monkeypatch)
    recall_turn(monkeypatch, "Last time you retrained logd.")
    bosco_mod._ask("lets continue")
    bosco_mod._ask("last session")  # a second recall doesn't carry the thread in twice

    record = stored(isolated_session_store)
    assert [ex["prompt"] for ex in record["exchanges"]] == [
        "retrain logd with scaffold split",
        "lets continue",
        "last session",
    ]
    assert [ex["worked"] for ex in record["exchanges"]] == [True, False, False]

    # Restart: the next REPL still names the real work
    bosco_utils.previous_session.cache_clear()
    assert '"retrain logd with scaffold split"' in bosco_utils.last_session_teaser()
