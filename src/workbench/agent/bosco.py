"""Bosco: the Workbench ML agent.

Runs Claude (via Bedrock) in the REPL with tools that execute against the
user's live session. Anything typed that isn't valid Python is routed here:

    what pxr models do we have?           # auto-routed
    bosco what models do we have          # explicit, for text that IS valid Python
"""

import re
import codeop
import keyword
import builtins
import logging
from contextlib import contextmanager

# Workbench Imports
from workbench.utils.repl_utils import cprint, Spinner, render_markdown
from workbench.utils.log_utils import log_level
from workbench.utils.bedrock_utils import claude_client, DEFAULT_MODEL
from workbench.agent.tools import (
    TOOL_SCHEMAS,
    dispatch,
    guide_index,
    general_guide,
    personality_text,
    DEFAULT_PERSONALITY,
)

log = logging.getLogger("workbench")

# The anthropic client logs every request at INFO, which buries Bosco's replies
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_TOKENS = 8000
MAX_TOOL_ROUNDS = 25  # bounds a single turn, not the conversation

# Every round of a turn resends the whole conversation, so an unbounded history
# costs quadratically over a session. Roughly 50k tokens.
MAX_HISTORY_CHARS = 200_000

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

## Voice

{personality}

The voice is a surface layer -- it never changes the work. Names, numbers, and
code stay exact, and the answer never gets buried under the act.

Guides available via read_guide:
{guides}

Read the relevant guide before building anything non-trivial, and whenever one
covers what the user is asking about. They are authoritative -- prefer them over
your own assumptions rather than answering from general knowledge."""


def _system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        general=general_guide().strip(),
        personality=personality_text(getattr(bosco, "personality", DEFAULT_PERSONALITY)).strip(),
        guides=guide_index() or "  (none)",
    )


def _text_of(message) -> str:
    return "\n".join(b.text for b in message.content if b.type == "text").strip()


@contextmanager
def _spinner(message: str):
    """Animate while Bosco waits, then erase the line so replies stay clean."""
    spinner = Spinner("lightpurple", message)
    spinner.start()
    try:
        yield
    finally:
        spinner.stop(clear=True)


def _namespace() -> dict:
    from IPython import get_ipython

    shell = get_ipython()
    return shell.user_ns if shell else globals()


def _history_chars() -> int:
    return sum(len(str(m["content"])) for m in _history)


def _trim_history() -> None:
    """Drop the oldest exchanges once the conversation gets large.

    Only cuts at a real user prompt, so a tool_use block is never separated from
    its tool_result -- the API rejects that pairing.
    """
    while _history_chars() > MAX_HISTORY_CHARS:
        cut = next(
            (
                i
                for i in range(1, len(_history))
                if _history[i]["role"] == "user" and isinstance(_history[i]["content"], str)
            ),
            None,
        )
        if cut is None:
            return  # nothing safe to drop
        del _history[:cut]


def _cached_messages() -> list:
    """History with a cache breakpoint on the newest message.

    Each round of a turn resends tools + system + the whole conversation, so the
    prefix is identical every time. One rolling breakpoint at the end lets all of
    it come back as a cache read instead of being re-billed.

    The last message is always a user message here (a prompt or tool results),
    which is why only those two content shapes need handling.
    """
    if not _history:
        return _history
    messages = list(_history)
    last = messages[-1]
    content = last["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    else:
        content = [dict(block) for block in content]
    content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
    messages[-1] = {**last, "content": content}
    return messages


def _run_turn(namespace: dict) -> None:
    """Send the current history, running tools until Claude is done."""
    global _client
    if _client is None:
        _client = claude_client()

    for _ in range(MAX_TOOL_ROUNDS):
        with _spinner("🐶  Bosco is thinking:"):
            response = _client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=MAX_TOKENS,
                system=_system_prompt(),
                tools=TOOL_SCHEMAS,
                messages=_cached_messages(),
            )

        text = _text_of(response)
        if text:
            render_markdown(text)

        _history.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "run_python" and bosco.show_code:
                cprint("grey", block.input.get("code", ""))
            with _spinner("🐶  Bosco is working:"):
                content = dispatch(block.name, block.input, namespace)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
        _history.append({"role": "user", "content": results})

    cprint("darkyellow", f"Stopped after {MAX_TOOL_ROUNDS} tool rounds.")


def _close_pending_tools(note: str) -> None:
    """Give any unanswered tool_use blocks a result.

    An interrupt can land after Claude asks for a tool but before we return the
    result. The API rejects that pairing on the next call, so the conversation
    would be stuck; closing them out keeps the history usable.
    """
    if not _history or _history[-1]["role"] != "assistant":
        return
    pending = [b for b in _history[-1]["content"] if getattr(b, "type", None) == "tool_use"]
    if pending:
        _history.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": b.id, "content": note} for b in pending],
            }
        )


def _ask(prompt: str) -> None:
    """One user turn against the shared history."""
    _history.append({"role": "user", "content": prompt})
    _trim_history()
    try:
        # Quiet routine INFO chatter from the code Bosco runs; restored afterwards
        with log_level():
            _run_turn(_namespace())
    except KeyboardInterrupt:
        _close_pending_tools("Interrupted by the user before this finished.")
        cprint("darkyellow", "Interrupted. (Ctrl-C again at the prompt to exit.)")
    except Exception as e:
        _close_pending_tools(f"Failed: {type(e).__name__}")
        cprint("red", f"{type(e).__name__}: {e}")


def bosco(prompt: str = None):
    """Chat with Bosco, the Workbench ML agent.

    Just type a question at the prompt -- anything that isn't valid Python is
    routed here. `bosco <question>` works too, for text that is valid Python.

    bosco.show_code = True        -> also echo the code Bosco runs
    bosco.personality = "pirate"  -> voice: chipper (default), professional, pirate
    """
    if prompt:
        _ask(prompt)
        return
    cprint("lightpurple", "🐶  Just ask -- type a question at the prompt.")
    cprint("grey", "(⌥ Option+Enter or Ctrl-J for a new line. bosco.show_code = True to see the code.)")


# Echo the code Bosco runs. Off by default; set True to follow along.
bosco.show_code = False

# The agent's voice: "chipper" (default), "professional", or "pirate".
bosco.personality = DEFAULT_PERSONALITY


# A line transformer routes plain English to Bosco so you never switch modes:
# whatever you type that isn't valid Python becomes a question. Explicit
# `bosco ...` still works; valid Python, magics, and help (`obj?`) run normally.
_BOSCO_LINE = re.compile(r"^\s*bosco(\s+(?P<rest>.*))?\s*$")

# CommandCompiler tells apart *invalid* Python (SyntaxError -> a Bosco question)
# from *incomplete* Python (returns None -> a multi-line block still being typed,
# which we must leave alone so IPython keeps collecting lines).
_compile = codeop.CommandCompiler()


def _defined(name: str) -> bool:
    """True if the name resolves in the REPL session or as a builtin."""
    return name in _namespace() or hasattr(builtins, name)


def _is_python(text: str) -> bool:
    """True if text is complete, valid Python."""
    try:
        return _compile(text, "<bosco>", "exec") is not None
    except (SyntaxError, ValueError, OverflowError):
        return False


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

    # Leave IPython special syntax (magics, shell) alone.
    if src[0] in "%!/,;@":
        return lines

    # A trailing `?` is IPython's help operator only when what precedes it is a
    # real object (`Model?`, `fs.query?`). People end questions with `?` too, so
    # anything that isn't a valid expression keeps its question mark and goes to
    # Bosco -- otherwise IPython answers with "Object `me` not found."
    if src.endswith("?"):
        return lines if _is_python(src.rstrip("?")) else [f"bosco({src!r})\n"]

    # A lone undefined name could only ever raise NameError, so it is a reply to
    # Bosco ("both", "yes", "metrics"), not code. Defined names still run.
    if src.isidentifier() and not keyword.iskeyword(src) and not _defined(src):
        return [f"bosco({src!r})\n"]

    try:
        if _compile(src, "<bosco>", "exec") is None:
            return lines  # incomplete block -> IPython keeps collecting
    except (SyntaxError, ValueError, OverflowError):
        return [f"bosco({src!r})\n"]  # not Python -> a question for Bosco
    return lines  # valid Python -> run it normally


def _install_newline_keys(shell) -> None:
    """Make Ctrl-J / Option+Enter insert a newline at the main REPL prompt.

    Terminals can't send a distinct Shift+Enter, so users map it to Ctrl-J
    (hex 0x0a). Ctrl-J otherwise behaves as Enter, which is what we are
    deliberately overriding.
    """
    app = getattr(shell, "pt_app", None)
    if app is None:  # simple prompt / no terminal
        return

    @app.key_bindings.add("c-j")
    @app.key_bindings.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")


def register():
    """Install the `bosco <text>` line transformer on the running shell."""
    from IPython import get_ipython

    shell = get_ipython()
    if shell and _bosco_transform not in shell.input_transformers_cleanup:
        shell.input_transformers_cleanup.append(_bosco_transform)
        _install_newline_keys(shell)
