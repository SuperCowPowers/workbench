"""Tools Bosco can call, and their schemas."""

import io
import logging
import contextlib
import traceback
from pathlib import Path
from typing import List

from workbench.utils import bosco_utils
from workbench.utils.bosco_utils import MAX_REPORT_CHARS
from workbench.utils.web_utils import EGRESS_MODE

GUIDES_DIR = Path(__file__).parent / "guides"
PERSONALITIES_FILE = Path(__file__).parent / "personalities.md"
EGRESS_FILE = Path(__file__).parent / "egress.md"
DEFAULT_PERSONALITY = "chipper"

# Always injected into the system prompt (not offered in the lazy-read menu).
ALWAYS_LOADED = {"general"}

# Tool output lands in history and is resent every round after, so keep it tight.
# Hitting this usually means filtering belonged in the query.
MAX_OUTPUT_CHARS = 4000


def guide_names() -> List[str]:
    """Names of the lazy-read best-practice guides (excludes always-loaded ones)."""
    return sorted(p.stem for p in GUIDES_DIR.glob("*.md") if p.stem not in ALWAYS_LOADED)


def guide_index() -> str:
    """Guide names with their one-line descriptions, for the system prompt.

    Names alone don't tell Claude what a guide covers, so it skips ones that
    would have answered the question. The description is the `> one-liner`
    under each guide's H1, so the index stays in sync with the files.
    """
    entries = []
    for path in sorted(GUIDES_DIR.glob("*.md")):
        if path.stem in ALWAYS_LOADED:
            continue
        head = path.read_text().splitlines()[:5]
        desc = next((line.lstrip("> ").strip() for line in head if line.startswith(">")), "")
        entries.append(f"  {path.stem:18} {desc}" if desc else f"  {path.stem}")
    return "\n".join(entries)


def general_guide() -> str:
    """The always-loaded general instructions, injected into the system prompt."""
    path = GUIDES_DIR / "general.md"
    return path.read_text() if path.exists() else ""


def _sections(path: Path) -> dict:
    """Map each `## name` header in a markdown file to its body text."""
    sections, current = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(body).strip() for name, body in sections.items()}


def personality_names() -> List[str]:
    """The selectable personality names."""
    return list(_sections(PERSONALITIES_FILE))


def personality_text(name: str) -> str:
    """Body of the selected personality, falling back to the default."""
    sections = _sections(PERSONALITIES_FILE)
    return sections.get(name) or sections.get(DEFAULT_PERSONALITY, "")


def egress_text() -> str:
    """The egress rules for the configured mode, injected into the system prompt."""
    return _sections(EGRESS_FILE).get(EGRESS_MODE, "")


# Where the conversation itself goes. The egress modes cover requests the agent
# makes; this covers the prompts, which carry whatever it has already read.
PROVIDER_EGRESS = {
    "bedrock": (
        "This conversation runs on Claude through Bedrock inside the user's own AWS "
        "account, so the prompts never leave it."
    ),
    "anthropic": (
        "This conversation runs on Claude through the Anthropic API on the user's own "
        "API key, so prompt content -- including anything you read from their files, "
        "dataframes, or artifacts -- leaves this machine. Say so plainly if they ask."
    ),
    "trial": (
        "This conversation runs on Claude through the SuperCowPowers trial proxy, which "
        "forwards to Bedrock with zero data retention. Prompt content -- including "
        "anything you read from their files, dataframes, or artifacts -- leaves this "
        "machine and reaches SuperCowPowers. Say so plainly if they ask, and mention "
        "that connecting their own AWS account keeps it in-house."
    ),
}


def provider_egress_text() -> str:
    """Where this session's conversation goes, for the system prompt."""
    from workbench.utils.llm_utils import llm_provider

    return PROVIDER_EGRESS.get(llm_provider(), "")


def read_guide(name: str) -> str:
    """Read a bundled guide by name."""
    path = GUIDES_DIR / f"{name}.md"
    if not path.exists():
        return f"No guide named '{name}'. Available: {', '.join(guide_names())}"
    return path.read_text()


# Loggers to watch during a run. The `workbench` logger sets `propagate = False`
# (it owns its own handlers), so a root handler alone would miss it — watch both.
_CAPTURED_LOGGERS = ("", "workbench")


class _CaptureHandler(logging.Handler):
    """Collect WARNING+ records emitted while Bosco's code runs.

    Workbench code often logs an error and returns an empty/None result rather
    than raising. Those log lines go to the handlers' original stdout, which
    `redirect_stdout` doesn't touch, so `run_python`'s buffer stays clean and
    Bosco never learns why the result was empty. This hands them back.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture_logs():
    """Attach a capture handler to the watched loggers for the duration."""
    handler = _CaptureHandler()
    loggers = [logging.getLogger(name) for name in _CAPTURED_LOGGERS]
    for lg in loggers:
        lg.addHandler(handler)
    try:
        yield handler
    finally:
        for lg in loggers:
            lg.removeHandler(handler)


def _format_captured(records: list) -> str:
    """Render captured records as `LEVEL logger: message`, repeats collapsed.

    A log-in-a-loop would otherwise flood the output; identical (level, logger,
    message) lines fold into one with a `(xN)` count, preserving first-seen order.
    """
    counts, order = {}, []
    for r in records:
        key = (r.levelname, r.name, r.getMessage())
        if key not in counts:
            order.append(key)
        counts[key] = counts.get(key, 0) + 1
    lines = []
    for level, name, message in order:
        suffix = f" (x{counts[(level, name, message)]})" if counts[(level, name, message)] > 1 else ""
        lines.append(f"{level} {name}: {message}{suffix}")
    return "\n".join(lines)


def run_python(code: str, namespace: dict) -> str:
    """Execute code in the REPL namespace and return stdout plus any error.

    The namespace is the live REPL session, so anything assigned here stays
    available to the user afterwards. WARNING+ log records emitted during the run
    are appended too — Workbench often logs a failure and returns empty rather
    than raising, and those lines never reach stdout.
    """
    buffer = io.StringIO()
    with _capture_logs() as captured:
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                exec(code, namespace)
        except Exception:
            buffer.write(traceback.format_exc())

    output = buffer.getvalue().strip()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(output)} chars total]"

    logged = _format_captured(captured.records)
    if logged:
        # Budget the log section separately so it can't starve real stdout/tracebacks.
        if len(logged) > MAX_OUTPUT_CHARS:
            logged = logged[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(logged)} chars total]"
        section = f"--- logged during execution (not stdout) ---\n{logged}"
        output = f"{output}\n\n{section}" if output else section

    return output or "(no output)"


TOOL_SCHEMAS = [
    {
        "name": "run_python",
        "description": (
            "Execute Python in the user's live Workbench REPL session. Workbench "
            "classes (DataSource, FeatureSet, Model, Endpoint, Meta, ...) are already "
            "imported. Variables you assign persist for the user. Use print() to see "
            "values -- only stdout is returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source to execute"}},
            "required": ["code"],
        },
    },
    {
        "name": "read_guide",
        "description": (
            "Read a Workbench best-practices guide. Read the relevant guide before "
            "building anything non-trivial -- they carry conventions that are not "
            "obvious from the API alone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Guide name, without .md"}},
            "required": ["name"],
        },
    },
    {
        "name": "save_session",
        "description": (
            "Save a report on where this session ended up, when the user asks. Distill: "
            "the goal, artifacts by name, what was concluded, what is still open. Not a "
            "transcript. Read the 'sessions' guide for the format."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short session name, e.g. 'logd-cleanup'"},
                "report": {"type": "string", "description": f"Report markdown, under {MAX_REPORT_CHARS} chars"},
            },
            "required": ["name", "report"],
        },
    },
    {
        "name": "read_session",
        "description": (
            "Recall a saved session report. Use the bare name for your own, or " "'<user>/<name>' for someone else's."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Session name, or '<user>/<name>'"}},
            "required": ["name"],
        },
    },
]


def dispatch(name: str, tool_input: dict, namespace: dict) -> str:
    """Run a tool by name and return its result as text."""
    if name == "run_python":
        return run_python(tool_input["code"], namespace)
    if name == "read_guide":
        return read_guide(tool_input["name"])
    if name == "read_session":
        return bosco_utils.read_session(tool_input["name"])
    if name == "save_session":
        try:
            return f"Saved to {bosco_utils.save_session(tool_input['name'], tool_input['report'])}"
        except ValueError as e:
            return str(e)
    return f"Unknown tool: {name}"
