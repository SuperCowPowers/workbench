"""workbench.endpoints — SageMaker endpoint runtime + client.

This subpackage is intentionally import-cheap: it pulls in only stdlib +
boto3/pandas/numpy/joblib so it remains usable inside SageMaker inference
containers that do not (and should not) have the full ``workbench[aws]`` /
``workbench[modeling]`` extras installed.

Two audiences live here:

* **Client-side** — :mod:`fast_inference`, :mod:`async_inference` are called
  by orchestration code that needs to invoke a deployed endpoint.
* **Server-side** — :mod:`inference`, :mod:`training_harness`, :mod:`uq_harness`
  run *inside* the endpoint container as part of the generated model script.

Anything imported lazily by the runtime (sklearn, scipy, rdkit, xgboost, etc.)
should stay lazy — see the CI smoke test that enforces this contract.
"""
