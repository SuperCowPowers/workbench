"""Bosco: the Workbench ML agent.

Runs Claude (via Bedrock) in the REPL with tools that execute against the
user's live session. Two ways to reach it, sharing one conversation:

    bosco what pxr models do we have?     # one-shot, stay in the REPL
    bosco                                 # open an interactive conversation
"""

import re
import codeop
import logging

# Workbench Imports
from workbench.utils.repl_utils import cprint, colors
from workbench.utils.bedrock_utils import claude_client, DEFAULT_MODEL
from workbench.utils.config_manager import ConfigManager
from workbench.agent.tools import TOOL_SCHEMAS, dispatch, guide_names, general_guide

log = logging.getLogger("workbench")

# The anthropic client logs every request at INFO, which buries Bosco's replies
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_TOKENS = 8000
MAX_TOOL_ROUNDS = 25  # bounds a single turn, not the conversation

# One conversation for the whole REPL session, shared by one-shot and chat
_history = []
_client = None

# The runtime frame lives here; tunable behavior lives in guides/general.md,
# injected below so a human can edit it without touching code.
SYSTEM_PROMPT = """You are Bosco, an ML engineering agent inside the Workbench REPL.
Workbench builds AWS SageMaker model pipelines: DataSource -> FeatureSet -> Model -> Endpoint.

You work in the user's live REPL session. Variables you create stay available to
them afterwards, so use clear names and tell them what you left behind.

{general}

Guides available via read_guide: {guides}
Read the relevant guide before building anything non-trivial."""


def _system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        general=general_guide().strip(),
        guides=", ".join(guide_names()) or "(none)",
    )


def _text_of(message) -> str:
    return "\n".join(b.text for b in message.content if b.type == "text").strip()


def _namespace() -> dict:
    from IPython import get_ipython

    shell = get_ipython()
    return shell.user_ns if shell else globals()


def _run_turn(namespace: dict) -> None:
    """Send the current history, running tools until Claude is done."""
    global _client
    if _client is None:
        _client = claude_client()

    for _ in range(MAX_TOOL_ROUNDS):
        response = _client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            system=_system_prompt(),
            tools=TOOL_SCHEMAS,
            messages=_history,
        )

        text = _text_of(response)
        if text:
            cprint("lightblue", text)

        _history.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "run_python":
                cprint("grey", block.input.get("code", ""))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": dispatch(block.name, block.input, namespace),
                }
            )
        _history.append({"role": "user", "content": results})

    cprint("darkyellow", f"Stopped after {MAX_TOOL_ROUNDS} tool rounds.")


def _ask(prompt: str) -> None:
    """One user turn against the shared history."""
    _history.append({"role": "user", "content": prompt})
    try:
        _run_turn(_namespace())
    except KeyboardInterrupt:
        cprint("darkyellow", "Interrupted.")
    except Exception as e:
        cprint("red", f"{type(e).__name__}: {e}")


def _prompt() -> str:
    """A colored '🐶 bosco:<config>>' prompt with the active AWS profile."""
    profile = ConfigManager().get_config("AWS_PROFILE") or "default"
    p, c, g, r = colors["lightpurple"], colors["darkyellow"], colors["grey"], colors["reset"]
    return f"\n🐶 {p}bosco{g}:{c}{profile}{p}>{r} "


def _chat() -> None:
    """Interactive conversation until the user leaves."""
    cprint("lightpurple", "🐶 Bosco — interactive mode. 'exit' to leave.")
    cprint("grey", "(Or skip this and just prefix a line: bosco <your question>)")
    prompt = _prompt()
    while True:
        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            break
        _ask(user_input)
    cprint("lightpurple", "Bosco out.")


def bosco(prompt: str = None):
    """Chat with Bosco, the Workbench ML agent.

    bosco("question")  -> one-shot answer
    bosco()            -> interactive conversation
    """
    if prompt:
        _ask(prompt)
    else:
        _chat()


# A line transformer routes plain English to Bosco so you never switch modes:
# whatever you type that isn't valid Python becomes a question. Explicit
# `bosco ...` still works; valid Python, magics, and help (`obj?`) run normally.
_BOSCO_LINE = re.compile(r"^\s*bosco(\s+(?P<rest>.*))?\s*$")

# CommandCompiler tells apart *invalid* Python (SyntaxError -> a Bosco question)
# from *incomplete* Python (returns None -> a multi-line block still being typed,
# which we must leave alone so IPython keeps collecting lines).
_compile = codeop.CommandCompiler()


def _bosco_transform(lines):
    if not lines:
        return lines
    src = "".join(lines).strip()
    if not src:
        return lines

    # Explicit `bosco ...` / bare `bosco` -> always route (even if valid Python)
    match = _BOSCO_LINE.match(lines[0].rstrip("\n"))
    if match:
        rest = (match.group("rest") or "").strip()
        if len(lines) > 1:
            rest = (rest + "\n" + "".join(lines[1:])).strip()
        if rest.startswith("="):  # `bosco = ...` is an assignment, not a question
            return lines
        return [f"bosco({rest!r})\n" if rest else "bosco()\n"]

    # Auto-route: only single, complete logical lines that aren't valid Python.
    # Leave multi-line cells and IPython special syntax (magics/shell/help) alone.
    if "\n" in src or src[0] in "%!/,;@" or src.endswith("?"):
        return lines
    try:
        if _compile(src, "<bosco>", "exec") is None:
            return lines  # incomplete block -> IPython keeps collecting
    except (SyntaxError, ValueError, OverflowError):
        return [f"bosco({src!r})\n"]  # not Python -> a question for Bosco
    return lines  # valid Python -> run it normally


def register():
    """Install the `bosco <text>` line transformer on the running shell."""
    from IPython import get_ipython

    shell = get_ipython()
    if shell and _bosco_transform not in shell.input_transformers_cleanup:
        shell.input_transformers_cleanup.append(_bosco_transform)
