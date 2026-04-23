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

from typing import List

from workbench.api.endpoint import Endpoint


class FeatureEndpoint(Endpoint):
    """Workbench Endpoint that reports its registered feature columns.

    Inherits async/realtime auto-routing from :class:`Endpoint` — for
    async-deployed endpoints, inference is transparently routed through an
    internal ``AsyncEndpoint`` so callers get correct behavior from a single
    object.
    """

    def feature_list(self) -> List[str]:
        """Return this endpoint's feature columns.

        Fast path: reads ``/workbench/feature_lists/<endpoint_name>`` from
        ParameterStore (populated by :func:`register_features` at deploy time).

        Freshness check: compares the parameter's ``LastModifiedDate`` to the
        endpoint's ``modified()`` time. If the endpoint has been redeployed
        since the feature list was cached, the cache is stale — we re-derive
        via the fallback path below and rewrite the cache. This means any
        ``to_endpoint()`` call is automatically picked up the next time
        ``feature_list()`` is called; no manual "remember to rerun
        register_features" step.

        Fallback (also used when there's no cache yet): runs a small smoke
        inference to discover the columns, writes them to ParameterStore so
        subsequent calls are fast, and returns the list. Uses 5 rows from the
        endpoint's input FeatureSet, subset to just the columns the model
        declares as inputs — see :func:`register_features` for the filtering
        rules (``desc*`` diagnostics and ``NON_FEATURE_COLUMNS`` provenance
        columns are excluded).

        Note on cost: the fallback triggers one inference call. For a realtime
        endpoint that's seconds; for a cold async endpoint it's the time to
        spin up an instance + run 5 rows (minutes). Only fires on miss or on
        endpoint-modified — every other call is just a ParameterStore lookup.

        Returns:
            List of feature column names.

        Raises:
            RuntimeError: If the fallback inference fails (e.g. no input model
                or input FeatureSet attached to this endpoint, or the endpoint
                produces no new columns — indicating it's not actually a
                feature endpoint).
        """
        from workbench.api.parameter_store import ParameterStore
        from workbench.utils.feature_endpoint_utils import feature_list_key, register_features

        ps = ParameterStore()
        key = feature_list_key(self.name)
        cols = ps.get(key)

        # Miss → derive and register
        if cols is None:
            self.log.important(
                f"FeatureEndpoint[{self.name}]: no feature list registered yet — "
                f"running smoke inference to discover and register columns."
            )
            return register_features(self)

        # Hit → is it fresh? Compare cache write-time vs endpoint modified-time.
        # If either timestamp is unavailable (transient AWS error, missing
        # metadata), fail open and trust the cache — staleness detection is
        # an optimization, not a correctness gate.
        param_modified = ps.last_modified(key)
        try:
            endpoint_modified = self.modified()
        except Exception:
            endpoint_modified = None

        if param_modified is not None and endpoint_modified is not None and endpoint_modified > param_modified:
            self.log.important(
                f"FeatureEndpoint[{self.name}]: endpoint modified at {endpoint_modified} "
                f"is newer than cached feature list ({param_modified}) — re-deriving."
            )
            return register_features(self)

        return cols
