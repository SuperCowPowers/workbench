"""Endpoint-import-smoke contract check.

Enforces the :mod:`workbench.endpoints` contract: every module under
``workbench/endpoints/`` must import cleanly against the leanest endpoint
dep manifest in the stack (``ci/endpoint_smoke_requirements.txt`` —
the intersection of every deployed endpoint container's requirements).
If a refactor moves a heavy transitive import (sagemaker, aiobotocore,
dash, cleanlab, umap-learn, matplotlib, mordred, ipython, networkx,
datasets, redis, cryptography, xgboost) into the import chain of any
``workbench.endpoints`` submodule, this script fails — telling us before
an endpoint deploy does.

The forbidden-heavy-deps blocklist is auto-derived from
``pyproject.toml``'s deps and extras minus
``ci/endpoint_smoke_requirements.txt`` — so adding a new pyproject
dep automatically extends the contract without hand-editing this script.

Invoked by :file:`tox.ini`'s ``endpoint-import-smoke`` env, which installs
against the lean smoke requirements file + workbench --no-deps before
running.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import workbench.endpoints

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_REQUIREMENTS = REPO_ROOT / "ci" / "endpoint_smoke_requirements.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Framework-specific modules — these live under workbench.endpoints but
# require their framework SDK at module-top. In a real SageMaker container
# the framework comes from the base image (pytorch_chem has torch+chemprop).
# The lean smoke env doesn't install them, so these are expected to
# ImportError here — that's fine *as long as* the only thing missing is the
# declared framework. If one of these modules fails for any OTHER reason
# (e.g. pulled in sagemaker), the smoke check still fails.
_FRAMEWORK_SPECIFIC = {
    "workbench.endpoints.pytorch_utils": {"torch"},
    "workbench.endpoints.chemprop_shap_utils": {"torch", "chemprop"},
    "workbench.endpoints.chemprop_utils": {"chemprop"},
}

# Extras that don't ship to any container at runtime and aren't relevant to
# the endpoint contract. Skipped when deriving the forbidden list.
_IGNORED_EXTRAS = {"dev"}

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _canon(name: str) -> str:
    return name.lower().replace("_", "-")


def _parse_name(line: str) -> str | None:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    m = _NAME_RE.match(line)
    return _canon(m.group(1)) if m else None


def _parse_requirements(path: Path) -> set[str]:
    return {n for n in (_parse_name(line) for line in path.read_text().splitlines()) if n}


def _parse_pyproject_packages() -> set[str]:
    """Every package referenced by pyproject.toml's deps + non-dev extras."""
    data = tomllib.loads(PYPROJECT.read_text())
    names: set[str] = set()
    for spec in data["project"]["dependencies"]:
        n = _parse_name(spec)
        if n and n != "workbench":
            names.add(n)
    for extra, specs in data["project"].get("optional-dependencies", {}).items():
        if extra in _IGNORED_EXTRAS:
            continue
        for spec in specs:
            n = _parse_name(spec)
            if n and n != "workbench":
                names.add(n)
    return names


def _is_installed(canonical_name: str) -> bool:
    """Check whether a pypi package is installed in the current env. Uses
    importlib.metadata so we don't have to maintain a pypi-name → import-
    name mapping (mordredcommunity → mordred, pyyaml → yaml, etc.)."""
    try:
        distribution(canonical_name)
        return True
    except PackageNotFoundError:
        return False


def _missing_framework(name: str, exc: BaseException) -> str | None:
    """If this module is framework-specific and the exception is about its
    missing framework, return the framework name. Otherwise None (real failure)."""
    if name not in _FRAMEWORK_SPECIFIC or not isinstance(exc, ModuleNotFoundError):
        return None
    missing = exc.name or ""
    for framework in _FRAMEWORK_SPECIFIC[name]:
        if missing == framework or missing.startswith(framework + "."):
            return framework
    return None


def check_endpoints_contract() -> list[tuple[str, BaseException]]:
    """Import every module under workbench.endpoints; return (name, exc) failures."""
    failures: list[tuple[str, BaseException]] = []
    discovered: list[str] = []
    for _, name, _ in pkgutil.iter_modules(workbench.endpoints.__path__, prefix="workbench.endpoints."):
        discovered.append(name)
        try:
            importlib.import_module(name)
            print(f"  OK   {name}")
        except BaseException as e:  # noqa: BLE001 — we want every kind of failure
            framework = _missing_framework(name, e)
            if framework is not None:
                print(f"  SKIP {name}: framework-specific (needs '{framework}')")
            else:
                print(f"  FAIL {name}: {type(e).__name__}: {e}")
                failures.append((name, e))
    print(f"\nworkbench.endpoints: discovered {len(discovered)} modules, {len(failures)} failure(s)")
    return failures


def check_no_heavy_deps_installed() -> list[str]:
    """Return any pyproject-listed package that's installed but isn't in the
    lean smoke requirements (i.e. shouldn't be reachable from an endpoint
    container's import surface)."""
    smoke_packages = _parse_requirements(SMOKE_REQUIREMENTS)
    forbidden = sorted(_parse_pyproject_packages() - smoke_packages)
    return [name for name in forbidden if _is_installed(name)]


def main() -> int:
    print("=== workbench.endpoints contract check ===")
    failures = check_endpoints_contract()

    print("\n=== heavy-dep exclusion check ===")
    print("(forbidden list = pyproject deps + non-dev extras MINUS ci/endpoint_smoke_requirements.txt)")
    leaked = check_no_heavy_deps_installed()
    if leaked:
        print(f"  LEAKED: {leaked}")
    else:
        print("  OK   no forbidden heavy deps are installed")

    if failures or leaked:
        print("\nendpoint-import-smoke FAILED", file=sys.stderr)
        return 1
    print("\nendpoint-import-smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
