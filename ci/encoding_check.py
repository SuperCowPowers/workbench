"""Explicit text-encoding check.

Text I/O without ``encoding=`` decodes with the platform locale encoding: UTF-8
on Linux/macOS, cp1252 on most Windows installs. Any non-ASCII byte in a file
Workbench reads (a guide, a config, a JSON artifact) then crashes Windows users
with ``UnicodeDecodeError``.

This check AST-scans ``src/workbench`` (``*.py`` and ``*.template``) and fails on:

* builtin ``open()`` in text mode without ``encoding=``
* ``.read_text()`` / ``.write_text()`` without ``encoding=``

``.read_text``/``.write_text`` exist only on path-like objects, so the attribute
name alone identifies them. ``.open()`` is left alone: ``tarfile.open``,
``Image.open`` and friends share the name. An ``open()`` whose mode isn't a
string literal is skipped, since it can't be judged statically.

Invoked by :file:`tox.ini`'s ``encoding-check`` env and the Python-lint workflow.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOT = REPO_ROOT / "src" / "workbench"

# Positional index of `encoding` in each signature: open(file, mode, buffering, encoding),
# read_text(encoding), write_text(data, encoding)
_ENCODING_POS = {"open": 3, "read_text": 0, "write_text": 1}


def _has_encoding(call: ast.Call, name: str) -> bool:
    if any(kw.arg == "encoding" or kw.arg is None for kw in call.keywords):
        return True
    return len(call.args) > _ENCODING_POS[name]


def _open_mode(call: ast.Call) -> ast.expr | None:
    """The mode argument of an open() call, or None when it's defaulted."""
    if len(call.args) > 1:
        return call.args[1]
    return next((kw.value for kw in call.keywords if kw.arg == "mode"), None)


def check_file(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, call)] for each text I/O call missing an explicit encoding."""
    violations = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            name = "open"
            mode = _open_mode(node)
            if mode is not None and not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)):
                continue
            if mode is not None and "b" in mode.value:
                continue
        elif isinstance(node.func, ast.Attribute) and node.func.attr in ("read_text", "write_text"):
            name = node.func.attr
        else:
            continue
        if not _has_encoding(node, name):
            violations.append((node.lineno, ast.unparse(node.func)))
    return violations


def main() -> int:
    print("=== explicit text-encoding check ===")
    print('(text I/O needs encoding="utf-8" — the Windows default is cp1252)')
    files = sorted(p for pattern in ("*.py", "*.template") for p in SCAN_ROOT.rglob(pattern))

    failures = 0
    for path in files:
        rel = path.relative_to(REPO_ROOT)
        for lineno, func in check_file(path):
            print(f"  FAIL {rel}:{lineno}: {func}() without encoding=")
            failures += 1

    print(f"\nscanned {len(files)} file(s), {failures} violation(s)")
    if failures:
        print("\nencoding-check FAILED", file=sys.stderr)
        return 1
    print("\nencoding-check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
