"""workbench.endpoints — THE import surface for SageMaker model scripts.

This subpackage is the contract: every symbol a generated model script
imports comes from ``workbench.endpoints.*``, full stop. The CI smoke test
(:file:`ci/endpoint_import_smoke.py`, run via the ``endpoint-import-smoke``
tox env) enumerates every module here and verifies they all import cleanly
against the leanest endpoint dep manifest (intersection of every deployed
endpoint container's requirements). Adding a new module under this directory
automatically extends the contract.

Three kinds of modules live here:

* **Real implementations** — :mod:`fast_inference`, :mod:`async_inference`,
  :mod:`inference`, :mod:`training_harness`, :mod:`uq_harness`,
  :mod:`pytorch_utils`, :mod:`chemprop_shap_utils`.
* **Re-exports** of code that lives at its "real" location in
  ``workbench.algorithms`` or ``workbench.utils`` — :mod:`uq_model_v0`,
  :mod:`uq_model_v1`, :mod:`uq_model_v2`, :mod:`uq_regression`,
  :mod:`fingerprint_proximity`, :mod:`proximity`, :mod:`residual_features`,
  :mod:`fingerprints`, :mod:`meta_endpoint_dag`. Keeps internal moves
  invisible to deployed model scripts.
* **Endpoint-friendly variants** of api classes — :mod:`df_store`,
  :mod:`parameter_store`. These auto-discover their config from the
  container env (env vars, Parameter Store, ambient IAM role) so model
  scripts can do ``DFStore()`` with zero args. The orchestration
  counterparts (:class:`workbench.api.DFStore`, etc.) require
  ``ConfigManager`` + ``AWSAccountClamp`` and are not endpoint-safe.

Framework-specific modules (:mod:`pytorch_utils`, :mod:`chemprop_shap_utils`)
are expected to require their framework's SDK; the smoke test allows-lists
them so they don't fail the lightweight check.
"""
