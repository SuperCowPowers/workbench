"""A declined request: Bosco retries it once on the fallback model; if that declines too,
Bosco says so and the turn leaves no trace in the conversation or the session record."""

from types import SimpleNamespace

import pytest

from workbench.agent import bosco as bosco_mod

PREFERRED = "claude-opus-5-5"
FALLBACK = bosco_mod.REFUSAL_FALLBACK
USAGE = SimpleNamespace(input_tokens=10, output_tokens=0)
BIO = SimpleNamespace(category="bio", explanation=None)


def reply(stop_reason, *content, details=None):
    return SimpleNamespace(usage=USAGE, content=list(content), stop_reason=stop_reason, stop_details=details)


def declined(details=BIO):
    """Both the preferred model and the fallback decline."""
    return [reply("refusal", details=details), reply("refusal", details=details)]


def text(words):
    return SimpleNamespace(type="text", text=words)


def tool_call():
    return SimpleNamespace(type="tool_use", id="toolu_1", name="note_topic", input={"title": "work", "new": False})


@pytest.fixture
def world(monkeypatch):
    """Bosco against scripted model responses."""
    world = SimpleNamespace(history=[], responses=[], sent=[], models=[], shown=[], recorded=[], model=PREFERRED)

    def stream(**kwargs):
        world.sent.append(kwargs["messages"])
        world.models.append(kwargs["model"])
        return world.responses.pop(0)

    monkeypatch.setattr(bosco_mod, "_history", world.history)
    monkeypatch.setattr(bosco_mod, "message_stream", stream)
    monkeypatch.setattr(bosco_mod, "model_id", lambda name: name)
    monkeypatch.setattr(bosco_mod, "_model", lambda: world.model)
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
    world.responses = declined(details)
    bosco_mod._ask("the question")
    assert expected in capsys.readouterr().out


def test_partial_output_is_not_shown_or_kept(world):
    world.responses = [reply("refusal", text("partial answer"), details=BIO), reply("refusal", details=BIO)]
    bosco_mod._ask("the question")
    assert world.shown == []
    assert world.history == []


def test_next_turn_does_not_resend_the_declined_prompt(world):
    world.responses = [reply("end_turn", text("first answer")), *declined(), reply("end_turn", text("third answer"))]
    bosco_mod._ask("first question")
    bosco_mod._ask("declined question")
    bosco_mod._ask("third question")

    assert "declined question" not in str(world.sent[-1])
    assert [m["role"] for m in world.sent[-1]] == ["user", "assistant", "user"]
    assert world.shown == ["first answer", "third answer"]


def test_refusal_mid_turn_drops_the_whole_turn(world):
    world.responses = [reply("end_turn", text("first answer")), reply("tool_use", tool_call()), *declined()]
    bosco_mod._ask("first question")
    kept = list(world.history)

    bosco_mod._ask("declined question")
    assert world.history == kept


def test_declined_turn_is_not_in_the_session_record(world):
    world.responses = [reply("end_turn", text("first answer")), *declined()]
    bosco_mod._ask("first question")
    bosco_mod._ask("declined question")
    assert world.recorded == ["first question"]


def test_fallback_answers_a_declined_turn(world, capsys):
    world.responses = [reply("refusal", text("partial answer"), details=BIO), reply("end_turn", text("the answer"))]
    bosco_mod._ask("the question")

    assert world.models == [PREFERRED, FALLBACK]
    assert world.sent[0] == world.sent[1]  # the same request, on the other model
    assert world.shown == ["the answer"]
    assert [m["role"] for m in world.history] == ["user", "assistant"]
    assert world.recorded == ["the question"]
    assert f"retrying on {FALLBACK}" in capsys.readouterr().out


def test_fallback_serves_the_rest_of_the_turn_only(world):
    world.responses = [
        reply("refusal", details=BIO),
        reply("tool_use", tool_call()),
        reply("end_turn", text("the answer")),
        reply("end_turn", text("next answer")),
    ]
    bosco_mod._ask("the question")
    bosco_mod._ask("next question")
    assert world.models == [PREFERRED, FALLBACK, FALLBACK, PREFERRED]


def test_no_retry_when_already_on_the_fallback(world, capsys):
    world.model = FALLBACK
    world.responses = [reply("refusal", details=BIO)]
    bosco_mod._ask("the question")

    assert world.models == [FALLBACK]
    assert "Claude declined this request" in capsys.readouterr().out
