"""FeatureEndpoint: Endpoint that knows its registered feature columns.

A *feature endpoint* takes an input column (usually ``smiles``) and emits
computed feature columns (RDKit/Mordred descriptors, fingerprints, 3D
Boltzmann features, etc.). This class adds a single new capability on top
of :class:`workbench.api.Endpoint`:

    from workbench.api import FeatureEndpoint

    fe = FeatureEndpoint("smiles-to-3d-full-v1")
    cols = fe.feature_list()    # list of 74 feature column names
    df_out = fe.inference(df_in) # full Endpoint API preserved

All other Endpoint behaviour is preserved — ``fe.inference(df)`` works
whether the underlying endpoint was deployed as realtime or async. For
async-deployed endpoints, FeatureEndpoint transparently routes inference
through SageMaker's async invocation path (S3 upload → invoke_async → poll)
so one class handles both transport types.

Deploy-time writer: :func:`workbench.utils.feature_endpoint_utils.register_features`
(called from the endpoint's deploy script) — populates the ParameterStore
entry that ``feature_list()`` reads back.
"""

from typing import List, Optional

from workbench.api.endpoint import Endpoint


class FeatureEndpoint(Endpoint):
    """Workbench Endpoint that reports its registered feature columns.

    Auto-detects whether the underlying endpoint was deployed as async or
    realtime (via ``workbench_meta["async_endpoint"]``). For async endpoints,
    inference is routed through an internal ``AsyncEndpoint`` so the
    S3-based async invocation path is used — callers get correct behavior
    from a single object.
    """

    def __init__(self, endpoint_name: str):
        super().__init__(endpoint_name)
        # If this endpoint was deployed async, keep an internal AsyncEndpoint
        # for inference delegation. All other Endpoint methods are inherited
        # from EndpointCore and work identically on either transport.
        if (self.workbench_meta() or {}).get("async_endpoint"):
            # Lazy import — keeps the api.endpoint ↔ api.async_endpoint import
            # graph clean (both end up importing EndpointCore).
            from workbench.api.async_endpoint import AsyncEndpoint

            self._async: Optional[AsyncEndpoint] = AsyncEndpoint(endpoint_name)
        else:
            self._async = None

    def feature_list(self) -> List[str]:
        """Return this endpoint's feature columns.

        Fast path: reads ``/workbench/feature_lists/<endpoint_name>`` from
        ParameterStore (populated by :func:`register_features` at deploy time).

        Fallback: if the ParameterStore entry is missing (e.g. the endpoint
        was deployed before the convention existed), runs a small smoke
        inference to discover the columns, writes them to ParameterStore so
        subsequent calls are fast, and returns the list. The smoke inference
        uses 5 rows from the endpoint's input FeatureSet, subset to just the
        columns the model declares as inputs — see :func:`register_features`
        for the filtering rules (``desc*`` diagnostics and ``NON_FEATURE_COLUMNS``
        provenance columns are excluded).

        Note on cost: the fallback path triggers one inference call. For a
        realtime endpoint that's seconds; for a cold async endpoint it's the
        time to spin up an instance + run 5 rows (minutes). Happens at most
        once per endpoint — subsequent calls hit ParameterStore.

        Returns:
            List of feature column names.

        Raises:
            RuntimeError: If the fallback inference fails (e.g. no input model
                or input FeatureSet attached to this endpoint, or the endpoint
                produces no new columns — indicating it's not actually a
                feature endpoint).
        """
        from workbench.utils.feature_endpoint_utils import get_endpoint_features, register_features

        cols = get_endpoint_features(self.name)
        if cols is None:
            self.log.important(
                f"FeatureEndpoint[{self.name}]: no feature list registered yet — "
                f"running smoke inference to discover and register columns."
            )
            cols = register_features(self)
        return cols

    # --------------------------------------------------------------------
    # Inference delegation — routes to the async path for async-deployed
    # endpoints. All Endpoint methods that do inference (auto_inference,
    # full_inference, ts_inference, cross_fold_inference, etc.) internally
    # call self.inference() / self.fast_inference(), so overriding just
    # these two routes the whole family correctly.
    # --------------------------------------------------------------------
    def inference(self, *args, **kwargs):
        if self._async is not None:
            return self._async.inference(*args, **kwargs)
        return super().inference(*args, **kwargs)

    def fast_inference(self, *args, **kwargs):
        if self._async is not None:
            return self._async.fast_inference(*args, **kwargs)
        return super().fast_inference(*args, **kwargs)
