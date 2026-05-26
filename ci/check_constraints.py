"""Constraints-coverage check.

Verifies every package referenced by:
  - pyproject.toml's `dependencies` + `optional-dependencies`
  - every ``requirements.txt`` under sagemaker_images/ and applications/

is pinned to an exact ``==`` version in constraints.txt at the repo root.
Prevents version drift between pyproject specs (what the pip-installer sees)
and the image-side manifests (what containers actually ship).

Invoked from .github/workflows/constraints-coverage.yml.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Package name pattern per PEP 508 (case-insensitive, normalize underscores to
# hyphens when comparing — pip treats `dash_ag_grid` and `dash-ag-grid` as
# the same package).
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_PIN_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*(\S+)\s*$")


def _canon(name: str) -> str:
    return name.lower().replace("_", "-")


def _parse_name(line: str) -> str | None:
    """Extract canonical package name from a requirement-file line. Returns
    None for comments, blanks, and pip option lines (e.g. ``--index-url``)."""
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    m = _NAME_RE.match(line)
    return _canon(m.group(1)) if m else None


def parse_requirements_file(path: Path) -> set[str]:
    return {n for n in (_parse_name(line) for line in path.read_text().splitlines()) if n}


def parse_pyproject() -> set[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    names: set[str] = set()
    for spec in data["project"]["dependencies"]:
        n = _parse_name(spec)
        if n and n != "workbench":
            names.add(n)
    for specs in data["project"].get("optional-dependencies", {}).values():
        for spec in specs:
            n = _parse_name(spec)
            if n and n != "workbench":
                names.add(n)
    return names


def parse_constraints() -> dict[str, str]:
    """Return ``{canonical-name: pinned-version}`` from constraints.txt."""
    pins: dict[str, str] = {}
    for line in (REPO_ROOT / "constraints.txt").read_text().splitlines():
        body = line.split("#", 1)[0].strip()
        if not body or body.startswith("-"):
            continue
        m = _PIN_RE.match(body)
        if m:
            pins[_canon(m.group(1))] = m.group(2)
    return pins


def find_deps_files() -> list[Path]:
    files: list[Path] = []
    for subdir in ("sagemaker_images", "applications"):
        root = REPO_ROOT / subdir
        if not root.exists():
            continue
        files.extend(root.rglob("requirements.txt"))
    return sorted(files)


def main() -> int:
    pins = parse_constraints()
    print(f"constraints.txt: {len(pins)} pinned packages")

    referenced: dict[str, set[Path]] = {}

    pyproject_path = REPO_ROOT / "pyproject.toml"
    for name in parse_pyproject():
        referenced.setdefault(name, set()).add(pyproject_path)

    for path in find_deps_files():
        for name in parse_requirements_file(path):
            referenced.setdefault(name, set()).add(path)

    print(f"referenced (across pyproject + image manifests): {len(referenced)} packages")

    missing = {n: locs for n, locs in referenced.items() if n not in pins}
    if missing:
        print(f"\nFAIL: {len(missing)} package(s) referenced without a constraints.txt pin")
        for name in sorted(missing):
            locs = ", ".join(str(p.relative_to(REPO_ROOT)) for p in sorted(missing[name]))
            print(f"  {name}  (referenced in: {locs})")
        return 1

    orphans = sorted(set(pins) - set(referenced))
    if orphans:
        print(f"\nWARN: {len(orphans)} constraint(s) with no consumer (consider removing)")
        for name in orphans:
            print(f"  {name}  (pinned: {pins[name]})")

    print("\nOK: every referenced package is pinned in constraints.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
