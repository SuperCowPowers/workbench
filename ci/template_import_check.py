"""Template import-invariant check.

A generated model script runs in **two** containers: the training image and the
(leaner) inference endpoint. The endpoint imports the whole script at load time
(to reach ``model_fn``/``predict_fn``) but never runs its ``__main__``.

``workbench.training.*`` modules may pull training-only deps (ray, optuna,
chemprop) that are **absent from the endpoint image**. So a template may import
``workbench.training.*`` only from inside ``__main__`` (a deferred/local import) —
never at module scope. A top-level ``workbench.training`` import would execute at
endpoint load and crash every endpoint for that model.

This check enforces that invariant: it scans every ``*.template`` under
``src/workbench/model_scripts/`` and fails on any **top-level** (column-0, i.e.
module-scope) import of ``workbench.training``. Templates carry ``{{placeholder}}``
tokens and aren't valid Python, so the check is line-based rather than AST-based.

Invoked by :file:`tox.ini`'s ``template-import-check`` env and the Python-lint
workflow.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_ROOT = REPO_ROOT / "src" / "workbench" / "model_scripts"

# A module-scope import of workbench.training: starts at column 0 (no leading
# whitespace) — a deferred import inside a function/__main__ is indented.
_TOP_LEVEL_TRAINING_IMPORT = re.compile(r"^(from|import)\s+workbench\.training(\.|\s|$)")


def check_template(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, line)] for each top-level workbench.training import."""
    violations = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if _TOP_LEVEL_TRAINING_IMPORT.match(line):
            violations.append((lineno, line.strip()))
    return violations


def main() -> int:
    print("=== template import-invariant check ===")
    print("(no top-level `import workbench.training` — defer it inside __main__)")
    templates = sorted(TEMPLATE_ROOT.rglob("*.template"))
    if not templates:
        print(f"  no templates found under {TEMPLATE_ROOT}", file=sys.stderr)
        return 1

    failures = 0
    for path in templates:
        rel = path.relative_to(REPO_ROOT)
        violations = check_template(path)
        if violations:
            for lineno, line in violations:
                print(f"  FAIL {rel}:{lineno}: top-level training import -> {line}")
            failures += len(violations)
        else:
            print(f"  OK   {rel}")

    print(f"\nscanned {len(templates)} template(s), {failures} violation(s)")
    if failures:
        print("\ntemplate-import-check FAILED", file=sys.stderr)
        return 1
    print("\ntemplate-import-check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
