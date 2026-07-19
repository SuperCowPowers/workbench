"""Tools Bosco can call, and their schemas."""

import io
import contextlib
import traceback
from pathlib import Path
from typing import List

GUIDES_DIR = Path(__file__).parent / "guides"

# Always injected into the system prompt (not offered in the lazy-read menu).
ALWAYS_LOADED = {"general"}

# Tool output lands in history and is resent every round after, so keep it tight.
# Hitting this usually means filtering belonged in the query.
MAX_OUTPUT_CHARS = 4000


def guide_names() -> List[str]:
    """Names of the lazy-read best-practice guides (excludes always-loaded ones)."""
    return sorted(p.stem for p in GUIDES_DIR.glob("*.md") if p.stem not in ALWAYS_LOADED)


def general_guide() -> str:
    """The always-loaded general instructions, injected into the system prompt."""
    path = GUIDES_DIR / "general.md"
    return path.read_text() if path.exists() else ""


def read_guide(name: str) -> str:
    """Read a bundled guide by name."""
    path = GUIDES_DIR / f"{name}.md"
    if not path.exists():
        return f"No guide named '{name}'. Available: {', '.join(guide_names())}"
    return path.read_text()


def run_python(code: str, namespace: dict) -> str:
    """Execute code in the REPL namespace and return stdout plus any error.

    The namespace is the live REPL session, so anything assigned here stays
    available to the user afterwards.
    """
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(code, namespace)
    except Exception:
        buffer.write(traceback.format_exc())

    output = buffer.getvalue().strip()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(output)} chars total]"
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
