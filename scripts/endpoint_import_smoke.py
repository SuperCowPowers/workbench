"""Endpoint-import-smoke contract check.

Enforces the :mod:`workbench.endpoints` contract: every module under
``workbench/endpoints/`` must import cleanly in the lightweight install
(no ``[aws]`` / ``[modeling]`` / ``[ui]`` extras). If a refactor moves
a heavy transitive import (sagemaker, aiobotocore, dash, cleanlab,
umap-learn, matplotlib, mordred, ipython) into the import chain of any
``workbench.endpoints`` submodule, this script fails — telling us
before an endpoint deploy does.

The test enumerates the directory dynamically — adding a new module
under ``workbench/endpoints/`` automatically extends the contract.

Invoked by :file:`tox.ini`'s ``endpoint-import-smoke`` env.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys

import workbench.endpoints

# Heavy deps that must NOT be installed in the endpoint-safe environment.
# These come from `[aws]` / `[modeling]` / `[ui]` extras. If any are present
# here, either the test environment installed extras it shouldn't have, or
# a workbench.endpoints module dragged one in transitively.
_FORBIDDEN_HEAVY_DEPS = [
    "sagemaker",
    "aiobotocore",
    "cleanlab",
    "umap",
    "dash",
    "matplotlib",
    "mordred",
]

# Framework-specific modules — these live under workbench.endpoints but require
# their framework SDK at module-top. In a real SageMaker container that
# framework comes from the base image (the pytorch image has torch, the
# chemprop image has chemprop+torch, etc.). The lightweight test venv
# doesn't install them, so these are expected to ImportError here — that's
# fine *as long as* the only thing missing is the declared framework.
#
# If one of these modules fails for any OTHER reason (e.g. pulled in
# sagemaker), the smoke check should still fail.
_FRAMEWORK_SPECIFIC = {
    "workbench.endpoints.pytorch_utils": {"torch"},
    "workbench.endpoints.chemprop_shap_utils": {"torch", "chemprop"},
}


def _missing_framework(name: str, exc: BaseException) -> str | None:
    """If this module is framework-specific and the exception is about its missing
    framework, return the framework name. Otherwise return None (real failure)."""
    if name not in _FRAMEWORK_SPECIFIC or not isinstance(exc, ModuleNotFoundError):
        return None
    missing = exc.name or ""
    # Match if the missing module is exactly the framework or a submodule of it.
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
    """Return any forbidden heavy deps that are importable in this environment."""
    return [m for m in _FORBIDDEN_HEAVY_DEPS if importlib.util.find_spec(m) is not None]


def main() -> int:
    print("=== workbench.endpoints contract check ===")
    failures = check_endpoints_contract()

    print("\n=== heavy-dep exclusion check ===")
    leaked = check_no_heavy_deps_installed()
    if leaked:
        print(f"  LEAKED: {leaked}")
    else:
        print(f"  OK   none of {_FORBIDDEN_HEAVY_DEPS} are installed")

    if failures or leaked:
        print("\nendpoint-import-smoke FAILED", file=sys.stderr)
        return 1
    print("\nendpoint-import-smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
