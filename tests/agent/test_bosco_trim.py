"""History trimming: a cut drops the oldest exchanges and every thinking block, since each
thinking block is bound to the exact conversation before it."""

from types import SimpleNamespace

import pytest

from workbench.agent import bosco as bosco_mod


def block(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


def exchange(prompt):
    """One finished turn: prompt, a tool round, and the answer, with thinking on each reply."""
    tool_use = block("tool_use", id=f"toolu_{prompt}", name="run_python", input={"code": "x"})
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": [block("thinking", thinking="", signature="sig"), tool_use]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "ok"}]},
        {"role": "assistant", "content": [block("redacted_thinking", data="x"), block("text", text="done")]},
    ]


def kinds(history):
    blocks = [b for m in history if not isinstance(m["content"], str) for b in m["content"]]
    return [b["type"] if isinstance(b, dict) else b.type for b in blocks]


@pytest.fixture
def history(monkeypatch):
    history = []
    monkeypatch.setattr(bosco_mod, "_history", history)
    return history


def test_cut_drops_oldest_exchange_and_all_thinking(history, monkeypatch):
    history.extend(exchange("first") + exchange("second") + [{"role": "user", "content": "third"}])
    monkeypatch.setattr(bosco_mod, "MAX_HISTORY_CHARS", bosco_mod._history_chars() - 1)

    bosco_mod._trim_history()

    assert history[0]["content"] == "second"
    assert "thinking" not in kinds(history) and "redacted_thinking" not in kinds(history)
    assert kinds(history) == ["tool_use", "tool_result", "text"]


def test_no_cut_keeps_thinking(history, monkeypatch):
    history.extend(exchange("first") + [{"role": "user", "content": "second"}])
    before = list(history)

    bosco_mod._trim_history()

    assert history == before


def test_message_left_empty_is_dropped(history, monkeypatch):
    only_thinking = {"role": "assistant", "content": [block("thinking", thinking="", signature="sig")]}
    history.extend(exchange("first") + [{"role": "user", "content": "second"}, only_thinking])
    history.append({"role": "user", "content": "third"})
    monkeypatch.setattr(bosco_mod, "MAX_HISTORY_CHARS", bosco_mod._history_chars() - 1)

    bosco_mod._trim_history()

    assert [m["content"] for m in history] == ["second", "third"]
