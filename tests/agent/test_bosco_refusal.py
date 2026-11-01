"""A declined request: Bosco says so, and the turn leaves no trace in the conversation or the session record."""

from types import SimpleNamespace

import pytest

from workbench.agent import bosco as bosco_mod

USAGE = SimpleNamespace(input_tokens=10, output_tokens=0)
BIO = SimpleNamespace(category="bio", explanation=None)


def reply(stop_reason, *content, details=None):
    return SimpleNamespace(usage=USAGE, content=list(content), stop_reason=stop_reason, stop_details=details)


def text(words):
    return SimpleNamespace(type="text", text=words)


def tool_call():
    return SimpleNamespace(type="tool_use", id="toolu_1", name="note_topic", input={"title": "work", "new": False})


@pytest.fixture
def world(monkeypatch):
    """Bosco against scripted model responses."""
    world = SimpleNamespace(history=[], responses=[], sent=[], shown=[], recorded=[])

    def stream(**kwargs):
        world.sent.append(kwargs["messages"])
        return world.responses.pop(0)

    monkeypatch.setattr(bosco_mod, "_history", world.history)
    monkeypatch.setattr(bosco_mod, "message_stream", stream)
    monkeypatch.setattr(bosco_mod, "model_id", lambda name: name)
    monkeypatch.setattr(bosco_mod, "job_updates", lambda prompt: prompt)
    monkeypatch.setattr(bosco_mod, "_namespace", lambda: {})
    monkeypatch.setattr(bosco_mod, "_system_prompt", lambda: "system")
    monkeypatch.setattr(bosco_mod, "render_markdown", world.shown.append)
    monkeypatch.setattr(bosco_mod, "_record_turn", lambda prompt, turn: world.recorded.append(prompt))
    monkeypatch.setattr(bosco_mod.bosco, "usage", dict.fromkeys(bosco_mod.bosco.usage, 0))
    return world


@pytest.mark.parametrize(
    "details, expected",
    [
        (None, "Claude declined this request. "),
        (SimpleNamespace(category=None, explanation=None), "Claude declined this request. "),
        (BIO, "Claude declined this request (bio). "),
        (SimpleNamespace(category="bio", explanation="Policy"), "Claude declined this request (Policy). "),
    ],
)
def test_user_is_told_why_when_the_api_says(world, capsys, details, expected):
    world.responses = [reply("refusal", details=details)]
    bosco_mod._ask("the question")
    assert expected in capsys.readouterr().out


def test_partial_output_is_not_shown_or_kept(world):
    world.responses = [reply("refusal", text("partial answer"), details=BIO)]
    bosco_mod._ask("the question")
    assert world.shown == []
    assert world.history == []


def test_next_turn_does_not_resend_the_declined_prompt(world):
    world.responses = [
        reply("end_turn", text("first answer")),
        reply("refusal", details=BIO),
        reply("end_turn", text("third answer")),
    ]
    bosco_mod._ask("first question")
    bosco_mod._ask("declined question")
    bosco_mod._ask("third question")

    assert "declined question" not in str(world.sent[-1])
    assert [m["role"] for m in world.sent[-1]] == ["user", "assistant", "user"]
    assert world.shown == ["first answer", "third answer"]


def test_refusal_mid_turn_drops_the_whole_turn(world):
    world.responses = [
        reply("end_turn", text("first answer")),
        reply("tool_use", tool_call()),
        reply("refusal", details=BIO),
    ]
    bosco_mod._ask("first question")
    kept = list(world.history)

    bosco_mod._ask("declined question")
    assert world.history == kept


def test_declined_turn_is_not_in_the_session_record(world):
    world.responses = [reply("end_turn", text("first answer")), reply("refusal", details=BIO)]
    bosco_mod._ask("first question")
    bosco_mod._ask("declined question")
    assert world.recorded == ["first question"]
