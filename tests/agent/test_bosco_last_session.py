"""The implicit last session: every Bosco turn rewrites a bounded record, grouped by the
topics Bosco names, and the next REPL greets the user with them.

The invariants: the record stays under the store's size budget however long the REPL
runs, a failed write never reaches the user, earlier topics shrink to their titles,
and once this REPL starts rewriting the record, the previous session is still what
Bosco reads.
"""

import json
from types import SimpleNamespace

from botocore.exceptions import TokenRetrievalError

from workbench.agent import bosco as bosco_mod
from workbench.agent.tools import dispatch
from workbench.repl.workbench_shell import BOSCO_INVITE, WorkbenchShell
from workbench.utils import bosco_utils


def text(value: str):
    return SimpleNamespace(type="text", text=value)


def call(name: str, **tool_input):
    return SimpleNamespace(type="tool_use", id=f"toolu_{name}", name=name, input=tool_input)


def scripted(monkeypatch, *turns):
    """Each _ask plays the next turn: a list of assistant content blocks (tool results are elided)."""
    queue = list(turns)

    def run_turn(namespace):
        for block in queue.pop(0):
            bosco_mod._history.append({"role": "assistant", "content": [block]})
            if block.type == "tool_use":
                bosco_mod._history.append({"role": "user", "content": [{"type": "tool_result"}]})

    monkeypatch.setattr(bosco_mod, "_history", [])
    monkeypatch.setattr(bosco_mod, "job_updates", lambda prompt: prompt)
    monkeypatch.setattr(bosco_mod, "_namespace", lambda: {})
    monkeypatch.setattr(bosco_mod, "_run_turn", run_turn)


def stored(store) -> dict:
    return store.get(bosco_utils.last_session_path(), warn=False)


def restart():
    """What a new REPL sees: the stored record, read fresh."""
    bosco_utils.previous_session.cache_clear()


def test_each_turn_saves(monkeypatch, isolated_session_store):
    scripted(monkeypatch, [text("You have 3 models.")], [call("run_python"), text("aqsol-v2 wins.")])
    bosco_mod._ask("what models do we have")
    bosco_mod._ask("which is best")

    assert stored(isolated_session_store)["exchanges"] == [
        {"prompt": "what models do we have", "reply": "You have 3 models.", "worked": False},
        {"prompt": "which is best", "reply": "aqsol-v2 wins.", "worked": True},
    ]


def test_new_topic_shrinks_the_old_one(monkeypatch, isolated_session_store):
    scripted(
        monkeypatch,
        [call("note_topic", title="logd retrain", new=True), call("run_python"), text("Started.")],
        [call("note_topic", title="logd retrain: done, MAE 0.41", new=False), text("Finished.")],
        [call("note_topic", title="PK curve metrics", new=True), call("run_python"), text("AUC added.")],
    )
    bosco_mod._ask("retrain logd with a scaffold split")
    bosco_mod._ask("how did it do")
    bosco_mod._ask("let's work on the PK curve metrics")

    record = stored(isolated_session_store)
    assert record["earlier"] == ["logd retrain: done, MAE 0.41"]
    assert record["topic"] == "PK curve metrics"
    assert [ex["prompt"] for ex in record["exchanges"]] == ["let's work on the PK curve metrics"]

    restart()
    assert bosco_utils.last_session_teaser() == (
        'Last time (just now): PK curve metrics (also: logd retrain: done, MAE 0.41) -- type "last session" to pick up'
    )


def test_recall_continues_the_thread(monkeypatch, isolated_session_store):
    scripted(monkeypatch, [call("note_topic", title="PK curve metrics", new=True), call("run_python"), text("AUC.")])
    bosco_mod._ask("let's work on the PK curve metrics")

    # Next REPL: "lets continue" then quit, twice over
    for _ in range(2):
        restart()
        monkeypatch.setattr(bosco_mod, "_window", bosco_utils.SessionWindow())
        scripted(monkeypatch, [call("read_last_session"), text("Last time: PK curve metrics.")])
        bosco_mod._ask("lets continue")

    record = stored(isolated_session_store)
    assert record["topic"] == "PK curve metrics"
    assert [ex["prompt"] for ex in record["exchanges"]] == [
        "let's work on the PK curve metrics",
        "lets continue",
        "lets continue",
    ]
    restart()
    assert bosco_utils.last_session_teaser().startswith("Last time (just now): PK curve metrics --")


def test_recall_after_starting_something_else(isolated_session_store):
    window = bosco_utils.SessionWindow()
    window.note_topic("aqsol comparison", new=True)
    window.continue_from({"earlier": ["logd retrain"], "topic": "PK curve metrics", "exchanges": []})
    assert window.earlier == ["logd retrain", "PK curve metrics"]
    assert window.topic == "aqsol comparison"


def test_record_stays_bounded(monkeypatch, isolated_session_store):
    turns = 100
    scripted(
        monkeypatch,
        *([call("note_topic", title=f"topic {i}", new=True), text("x" * 2000)] for i in range(turns)),
    )
    for i in range(turns):
        bosco_mod._ask(f"question {i} " + "y" * 2000)

    record = stored(isolated_session_store)
    assert len(json.dumps(record)) <= bosco_utils.LAST_SESSION_CHARS
    assert len(record["earlier"]) == bosco_utils.MAX_EARLIER
    assert record["topic"] == "topic 99"
    assert record["exchanges"][-1]["prompt"].startswith("question 99 ")


def test_failed_turn_is_still_recorded(monkeypatch, isolated_session_store):
    scripted(monkeypatch)

    def run_turn(namespace):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(bosco_mod, "_run_turn", run_turn)
    bosco_mod._ask("deploy it")
    assert stored(isolated_session_store)["exchanges"] == [{"prompt": "deploy it", "reply": "", "worked": False}]


def test_failed_write_stays_quiet(monkeypatch, isolated_session_store, capsys):
    def expired(name, value):
        raise TokenRetrievalError(provider="sso", error_msg="Token has expired and refresh failed")

    monkeypatch.setattr(isolated_session_store, "upsert", expired)
    scripted(monkeypatch, [text("You have 3 models.")])
    bosco_mod._ask("what models do we have")
    assert "TokenRetrievalError" not in capsys.readouterr().out


def test_previous_session_survives_this_repls_saves(monkeypatch, isolated_session_store):
    earlier = bosco_utils.SessionWindow()
    earlier.note_topic("logd retrain", new=True)
    earlier.add("train a logd model", "Started logd-xgb-v1.", worked=True)
    earlier.save()

    scripted(monkeypatch, [text("Hello again.")])
    bosco_mod._ask("hi")

    assert stored(isolated_session_store)["exchanges"][-1]["prompt"] == "hi"
    previous = dispatch("read_last_session", {}, {})
    assert "Latest topic: logd retrain" in previous
    assert "train a logd model" in previous and "Started logd-xgb-v1." in previous
    assert "**User:** hi" not in previous


def test_no_previous_session(isolated_session_store):
    assert dispatch("read_last_session", {}, {}) == "No previous session recorded."
    assert bosco_utils.last_session_teaser() is None
    assert WorkbenchShell.bosco_greeting() == BOSCO_INVITE


def test_teaser_without_topics_names_the_last_real_request(isolated_session_store):
    window = bosco_utils.SessionWindow()
    window.add("retrain logd with scaffold split", "Done.", worked=True)
    window.add("lets continue", "We were retraining logd.", worked=False)
    window.add("thanks", "Anytime!", worked=False)
    window.save()
    assert '"retrain logd with scaffold split"' in bosco_utils.last_session_teaser()
    assert bosco_utils.last_session_teaser() in WorkbenchShell.bosco_greeting()
