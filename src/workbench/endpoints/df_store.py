"""Endpoint-friendly :class:`DFStore` — auto-discovers config from the runtime.

This is the endpoint variant of :class:`workbench.api.DFStore`. Where the
api class pulls its bucket/session from :class:`ConfigManager` +
:class:`AWSAccountClamp` (orchestration deps, refreshable creds), this
variant auto-discovers everything from what's available in the container:

* ``s3_bucket`` ← ``WORKBENCH_BUCKET`` env var → Parameter Store at
  ``/workbench/config/workbench_bucket``
* ``boto3_session`` ← ambient SageMaker execution role via
  :func:`get_boto3_session`

Endpoint code can do:

    from workbench.endpoints.df_store import DFStore
    ds = DFStore()                          # zero args
    df = ds.get("/some/cached/data")

If both the env var and Parameter Store lookup fail, ``DFStore()`` raises
``ValueError`` — caller can pass an explicit ``s3_bucket`` to override.
"""

from workbench.core.df_store_core import DFStoreCore as DFStore  # noqa: F401

__all__ = ["DFStore"]
