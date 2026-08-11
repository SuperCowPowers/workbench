"""The line transformer that decides whether a cell runs as Python or asks Bosco.

Every keystroke in the REPL goes through `_bosco_transform`, and both failure
directions are bad: English executed as code is a confusing traceback, and code
shipped to Bosco is a Bedrock turn the user didn't ask for. This pins the boundary.
"""

import ast
import pytest

from workbench.agent import bosco as bosco_mod
from workbench.agent.bosco import _bosco_transform


@pytest.fixture(autouse=True)
def repl_namespace(monkeypatch):
    """A predictable REPL session, since `_defined` decides several of these cases."""
    namespace = {"df": object(), "model": object(), "models": lambda: None}
    monkeypatch.setattr(bosco_mod, "_namespace", lambda: namespace)
    return namespace


def routed(src: str):
    """The prompt handed to Bosco, or None if the cell stays Python.

    Args:
        src (str): What the user typed, newlines and all.

    Returns:
        str: The prompt Bosco is asked, or None when the cell is left for IPython.
    """
    lines = [f"{line}\n" for line in src.split("\n")]
    out = _bosco_transform(lines)
    if out == lines:
        return None
    call = ast.parse(out[0], mode="eval").body
    assert isinstance(call, ast.Call) and call.func.id == "bosco", f"unexpected rewrite: {out}"
    return call.args[0].value if call.args else ""


ENGLISH = [
    "what models do we have",
    "compare aqsol-v1 and aqsol-v2",
    "what's the mae for that endpoint",  # apostrophe: an unterminated string, not incomplete input
    "don't deploy it yet",
    "no thanks",
    "run it",
    "- look at the metrics",
    "1. check the model",
]


@pytest.mark.parametrize("src", ENGLISH)
def test_english_goes_to_bosco(src):
    assert routed(src) == src


PYTHON = [
    "df.head()",
    "model = 42",
    "for i in range(3):\n    print(i)",
    "models",  # defined in the session, so it runs
    "id",  # a builtin counts as defined
    "pass",  # a keyword is never mistaken for a reply
    "2 + 2",
]


@pytest.mark.parametrize("src", PYTHON)
def test_python_runs_normally(src):
    assert routed(src) is None


def test_incomplete_block_is_left_for_ipython():
    # Returning anything but the original lines here strands the user mid-block.
    assert _bosco_transform(["for i in range(3):\n"]) == ["for i in range(3):\n"]
    assert _bosco_transform(["endpoint.inference(df\n"]) == ["endpoint.inference(df\n"]


class TestRepliesToBosco:
    """An expression naming nothing that exists could only ever raise NameError."""

    @pytest.mark.parametrize("src", ["both", "yes", "sure", "metrics"])
    def test_lone_undefined_name(self, src):
        assert routed(src) == src

    @pytest.mark.parametrize("src", ["not sure", "not really", "yes or no", "aqsol or pxr"])
    def test_short_phrase_that_compiles(self, src):
        assert routed(src) == src

    @pytest.mark.parametrize("src", ["models", "id", "df", "not df"])
    def test_a_defined_name_anywhere_means_code(self, src):
        assert routed(src) is None

    @pytest.mark.parametrize("src", ["nope()", "nope.metrics", "nope[0]"])
    def test_invocation_shaped_stays_code(self, src):
        # A typo'd call belongs in an immediate NameError, not an agent turn.
        assert routed(src) is None

    @pytest.mark.parametrize("src", ["2 + 2", "[1, 2, 3]", "'a string'"])
    def test_expressions_with_no_names_run(self, src):
        assert routed(src) is None


class TestExplicitBosco:
    """`bosco <text>` forces a question even when the text is valid Python."""

    def test_forces_routing_of_valid_python(self):
        assert routed("bosco models") == "models"

    def test_bare_bosco_takes_no_prompt(self):
        assert routed("bosco") == ""

    def test_joins_continuation_lines(self):
        assert routed("bosco why did\nthat fail") == "why did\nthat fail"

    def test_assignment_is_not_a_question(self):
        assert routed("bosco = 5") is None

    def test_attribute_set_is_not_a_question(self):
        assert routed("bosco.show_code = True") is None


class TestIPythonSyntax:
    """Magics, shell escapes, and help stay with IPython."""

    @pytest.mark.parametrize("src", ["%time df.head()", "!ls -l", "%%bash\necho hi"])
    def test_specials_pass_through(self, src):
        assert routed(src) is None

    @pytest.mark.parametrize("src", ["?Model", "??Model", "?df.head"])
    def test_prefix_help_operator_passes_through(self, src):
        assert routed(src) is None


class TestTrailingQuestionMark:
    """`?` is never valid Python, so a line ending in one is always a question."""

    @pytest.mark.parametrize(
        "src",
        [
            "does that make sense?",
            "why?",  # a bare name, where the postfix help operator used to win
            "why is that?",  # `is` makes this compile, which used to route it to IPython
            "Model?",  # even a real object: `?Model` is how you ask for the docstring
            "df.head?",
        ],
    )
    def test_goes_to_bosco(self, src):
        assert routed(src) == src


def test_empty_cell_is_untouched():
    assert _bosco_transform([]) == []
    assert _bosco_transform(["\n"]) == ["\n"]
    assert _bosco_transform(["   \n"]) == ["   \n"]
