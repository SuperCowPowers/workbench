"""Shared utilities for Workbench model scripts.

This package is symlinked into each per-framework script bundle directory
(e.g. workbench/model_scripts/xgb_model/model_script_utils -> ../../model_script_utils)
so the SageMaker training and inference containers can import these helpers
without having the workbench package installed.

Re-exports the most commonly-used helpers from `core.py` at the package level so
flat-style imports like::

    from model_script_utils import compress_std_outliers, input_fn, output_fn

continue to work when this package is symlinked as `model_script_utils/` into a
script bundle.

Module-specific helpers should be imported from their submodules::

    from model_script_utils.uq_harness import compute_vgmu_confidence
    from model_script_utils.uq_model import UQModel
    from model_script_utils.fingerprint_proximity import FingerprintProximity

Note: the inner module is `core.py`, not `model_script_utils.py`. A
`model_script_utils/model_script_utils.py` file would shadow the package name and
trigger import resolution oddities in some environments (uvicorn workers in the
SageMaker inference container were one observed casualty).
"""

from .core import *  # noqa: F401, F403
