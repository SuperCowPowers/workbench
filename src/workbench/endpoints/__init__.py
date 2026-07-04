"""workbench.endpoints — the inference-time import surface for model scripts.

This subpackage is the contract for everything a generated model script imports
at **inference** time (``model_fn``/``predict_fn`` and what they reach).
Training-only code lives in :mod:`workbench.training` and is imported *only*
inside a template's ``__main__`` (the endpoint never runs ``__main__``, and the
training-only deps are absent from the endpoint image). The CI smoke test
(:file:`ci/endpoint_import_smoke.py`, run via the ``endpoint-import-smoke``
tox env) enumerates every module here and verifies they all import cleanly
against the leanest endpoint dep manifest (intersection of every deployed
endpoint container's requirements). Adding a new module under this directory
automatically extends the contract.

Three kinds of modules live here:

* **Real implementations** — :mod:`fast_inference`, :mod:`async_inference`,
  :mod:`inference`, :mod:`uq_harness`, :mod:`pytorch_utils`,
  :mod:`chemprop_shap_utils`.
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
