"""Tools Bosco can call, and their schemas."""

import io
import logging
import contextlib
import traceback
from pathlib import Path
from typing import List

GUIDES_DIR = Path(__file__).parent / "guides"
PERSONALITIES_FILE = Path(__file__).parent / "personalities.md"
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


def _personality_sections() -> dict:
    """Map each `## name` header in personalities.md to its body text."""
    sections, current = {}, None
    for line in PERSONALITIES_FILE.read_text().splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(body).strip() for name, body in sections.items()}


def personality_names() -> List[str]:
    """The selectable personality names."""
    return list(_personality_sections())


def personality_text(name: str) -> str:
    """Body of the selected personality, falling back to the default."""
    sections = _personality_sections()
    return sections.get(name) or sections.get(DEFAULT_PERSONALITY, "")


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
]


def dispatch(name: str, tool_input: dict, namespace: dict) -> str:
    """Run a tool by name and return its result as text."""
    if name == "run_python":
        return run_python(tool_input["code"], namespace)
    if name == "read_guide":
        return read_guide(tool_input["name"])
    return f"Unknown tool: {name}"
