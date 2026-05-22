"""Endpoint-friendly :class:`ParameterStore` — uses ambient container creds.

This is the endpoint variant of :class:`workbench.api.ParameterStore`. Where
the api class wraps with :class:`AWSAccountClamp` (refreshable creds), this
variant uses :func:`get_boto3_session` — which in service contexts
short-circuits to the container's attached IAM role.

Endpoint code can do:

    from workbench.endpoints.parameter_store import ParameterStore
    ps = ParameterStore()                      # zero args
    bucket = ps.get("/workbench/config/workbench_bucket")
"""

from workbench.core.parameter_store_core import ParameterStoreCore as ParameterStore  # noqa: F401

__all__ = ["ParameterStore"]
