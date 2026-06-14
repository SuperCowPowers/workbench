"""Dependency-creep guard for the workbench.lambda_layer subpackage.

Everything under workbench.lambda_layer ships in the published workbench Lambda
layer, whose bundled third-party deps are networkx and pandas (boto3/botocore are
provided by the Lambda runtime). This test pins that budget: it imports every
module in the subpackage with all non-allowlisted top-level imports blocked, so
adding a module that drags a new dependency fails here -- loudly, naming the
offender -- instead of at deploy time with a CloudWatch ``No module named ...``.

To intentionally grow the layer's footprint, add the dependency to ALLOWED below
(and to the layer build's requirements).
"""

import subprocess
import sys

import pytest

# The layer's dependency budget. Stdlib is always allowed (resolved at runtime
# via sys.stdlib_module_names). These are the non-stdlib top-level packages the
# layer code may import:
ALLOWED = {
    "workbench",  # internal (the subpackage and its siblings it reaches)
    "networkx",  # bundled third-party dep
    "pandas",  # bundled third-party dep (pulls numpy/dateutil/pytz transitively)
    "boto3",  # provided by the Lambda runtime
    "botocore",  # provided by the Lambda runtime
}

# Runs in a clean subprocess: install a meta-path finder that rejects any
# top-level import outside the budget, then import every lambda_layer module. A
# rejected import raises ModuleNotFoundError tagged "CREEP:<pkg>"; an *uncaught*
# one (i.e. a module that genuinely needs the dep) is reported via a clean
# CREEP_DEP= marker. (awswrangler is rejected too but __init__ catches it, so it
# never surfaces -- exactly the optional-dependency behavior we want.)
_GUARD = """
import importlib
import pkgutil
import sys

ALLOWED = set(%r) | set(sys.stdlib_module_names)


class CreepBlocker:
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top not in ALLOWED:
            raise ModuleNotFoundError(f"CREEP:{top}")
        return None  # allowed -> defer to the real finders


sys.meta_path.insert(0, CreepBlocker())

try:
    import workbench.lambda_layer as pkg

    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod.name)
except ModuleNotFoundError as e:
    if str(e).startswith("CREEP:"):
        print("CREEP_DEP=" + str(e).split("CREEP:", 1)[1], file=sys.stderr)
        sys.exit(2)
    raise

print("OK")
"""


def test_lambda_layer_has_no_dependency_creep():
    result = subprocess.run(
        [sys.executable, "-c", _GUARD % (sorted(ALLOWED),)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        offender = next(
            (line.split("CREEP_DEP=", 1)[1].strip() for line in result.stderr.splitlines() if "CREEP_DEP=" in line),
            None,
        )
        if offender:
            pytest.fail(
                f"workbench.lambda_layer imports {offender!r}, which is outside the layer's "
                f"dependency budget {sorted(ALLOWED)}. Either drop the dependency or, if it's "
                f"intentional, add it to ALLOWED here and to the layer build's requirements.\n\n"
                f"{result.stderr}"
            )
        pytest.fail(f"dependency guard failed unexpectedly:\n{result.stdout}\n{result.stderr}")

    assert "OK" in result.stdout
