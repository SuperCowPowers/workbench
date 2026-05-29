"""AsyncEndpointCore: Workbench Async Endpoint support.

Extends EndpointCore to support SageMaker async inference endpoints.
Async endpoints accept the same model artifacts and container images as
realtime endpoints, but invocations are non-blocking: input is uploaded
to S3, the response is written to an S3 output location, and the caller
polls for completion.

This is useful for workloads where per-invocation latency exceeds the
realtime 60-second server-side timeout (e.g., Boltzmann 3D descriptor
generation that can take minutes per molecule).

The API surface is identical to EndpointCore — ``inference()`` and
``fast_inference()`` return DataFrames synchronously, hiding the async
S3 round-trip from the caller.

Implementation: the protocol-level invocation lives in
``workbench.endpoints.async_inference``; this class adds
Workbench-specific concerns (``workbench_meta`` knobs for batch sizing
and concurrency, capture/monitoring, S3 path resolution).
"""

import logging

import pandas as pd

from workbench.core.artifacts.endpoint_core import EndpointCore
from workbench.endpoints.async_inference import async_inference, purge_async_queue

log = logging.getLogger("workbench")

# Default rows per invocation. Smaller batches give better load balancing
# across workers — a handful of slow rows in one chunk stretches total time
# less when there are more chunks to absorb the variance. At ~20s/row (typical
# async workload) the extra per-chunk overhead (~3s polling startup) is <2%.
# Fast endpoints (sub-second per row) should override higher via meta so the
# overhead doesn't dominate.
# Override per-endpoint via workbench_meta["inference_batch_size"].
_DEFAULT_BATCH_SIZE = 10

# Safety cap on client-side thread-pool size for direct (non-InferenceCache)
# calls with large DataFrames. Prevents thread-pool blowup on calls like
# ``end.inference(huge_df)``. Override per-call via
# workbench_meta["inference_max_in_flight"].
_MAX_IN_FLIGHT_CAP = 64

# Pandas option applied once at import — avoid mutating global state per call.
pd.set_option("future.no_silent_downcasting", True)


def _resolve_max_in_flight(meta: dict, n_batches: int) -> int:
    """Size the client-side thread pool for one ``_async_batch_invoke`` call.

    Default: ``n_batches`` — one worker per sub-batch, fully parallel, no
    queueing in the client pool.

    Safety cap: ``inference_max_in_flight`` from meta, defaulting to
    :data:`_MAX_IN_FLIGHT_CAP`. Prevents thread-pool blowup on calls with
    very large DataFrames.
    """
    cap = int(meta.get("inference_max_in_flight", _MAX_IN_FLIGHT_CAP))
    return min(n_batches, cap)


class AsyncEndpointCore(EndpointCore):
    """EndpointCore subclass for SageMaker async inference endpoints.

    Overrides the invocation path (_predict / fast_inference) to use the
    async S3 upload → invoke_async → poll S3 → download pattern.  All
    metadata, metrics, and capture logic is inherited unchanged.
    """

    def __init__(self, endpoint_name: str, **kwargs):
        super().__init__(endpoint_name, **kwargs)

    # -----------------------------------------------------------------
    # Override: _predict  (called by EndpointCore.inference for modeled endpoints)
    # -----------------------------------------------------------------
    def _predict(self, eval_df: pd.DataFrame, features: list[str], drop_error_rows: bool = False) -> pd.DataFrame:
        """Run async prediction on a DataFrame.

        Follows the same contract as EndpointCore._predict: accepts a
        DataFrame, returns a DataFrame with prediction/feature columns added.
        Internally uploads chunks to S3, calls invoke_async, polls for output.
        """
        if eval_df.empty:
            log.warning("Evaluation DataFrame has 0 rows.")
            return pd.DataFrame(columns=eval_df.columns)

        # Validate features
        df_columns_lower = set(col.lower() for col in eval_df.columns)
        features_lower = set(f.lower() for f in features)
        if not features_lower.issubset(df_columns_lower):
            missing = features_lower - df_columns_lower
            raise ValueError(f"DataFrame does not contain required features: {missing}")

        return self._async_batch_invoke(eval_df)

    # -----------------------------------------------------------------
    # Override: fast_inference  (called for "floating" endpoints with no model)
    # -----------------------------------------------------------------
    def fast_inference(self, eval_df: pd.DataFrame, threads: int = 4) -> pd.DataFrame:
        """Async version of fast_inference — ignores threads, uses S3 polling."""
        if eval_df.empty:
            return pd.DataFrame(columns=eval_df.columns)

        return self._async_batch_invoke(eval_df)

    # -----------------------------------------------------------------
    # Queue management
    # -----------------------------------------------------------------
    def purge_async_queue(self) -> int:
        """Cancel all queued async invocations for this endpoint.

        Thin wrapper over :func:`workbench.endpoints.async_inference.purge_async_queue`.
        See that function for behavior, caveats, and return semantics.
        """
        return purge_async_queue(
            endpoint_name=self.name,
            s3_bucket=self.workbench_bucket,
            sm_session=self.boto3_session,
        )

    # -----------------------------------------------------------------
    # Override: test_inference  (smaller default sample)
    # -----------------------------------------------------------------
    def test_inference(self, num_rows: int = 10) -> pd.DataFrame:
        """Smoke-test this async endpoint on a small sample.

        Async workloads can run at seconds-to-minutes per row, so the sample is
        capped low by default — enough to verify the endpoint responds end-to-end.

        Args:
            num_rows (int): Max number of rows to sample (default 10).

        Returns:
            pd.DataFrame: The inference results (empty if no model/data).
        """
        return super().test_inference(num_rows=num_rows)

    # -----------------------------------------------------------------
    # Internal: delegate to the lightweight bridges client
    # -----------------------------------------------------------------
    def _async_batch_invoke(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Delegate batch invocation to ``workbench.endpoints.async_inference``.

        Reads two tunable knobs from ``workbench_meta()``:
          * ``inference_batch_size`` (default 10): rows per invocation.
          * ``inference_max_in_flight`` (default :data:`_MAX_IN_FLIGHT_CAP`):
            outstanding invocation cap for direct calls bypassing
            :class:`InferenceCache`.

        For MetaEndpoints (detected via ``workbench_meta["meta_endpoint_dag"]``),
        an ``instances_str_fn`` callable is passed to ``async_inference`` so
        its ``instances=`` log field renders per-child counts instead of the
        meta orchestrator's own (always-1) count. The callable composes
        :meth:`Endpoint.instance_counts` per async child.
        """
        meta = self.workbench_meta() or {}
        batch_size = int(meta.get("inference_batch_size", _DEFAULT_BATCH_SIZE))
        # Estimate chunks for sizing; bridges re-derives this internally too.
        n_batches = max(1, (len(eval_df) + batch_size - 1) // batch_size)
        max_in_flight = _resolve_max_in_flight(meta, n_batches=n_batches)

        return async_inference(
            endpoint_name=self.endpoint_name,
            eval_df=eval_df,
            sm_session=self.boto3_session,
            batch_size=batch_size,
            max_in_flight=max_in_flight,
            s3_bucket=self.workbench_bucket,
            s3_input_prefix=f"endpoints/{self.name}/async-input",
            instances_str_fn=self._meta_instances_str_fn(meta),
        )

    def _meta_instances_str_fn(self, meta: dict):
        """Build the ``instances_str_fn`` callable for a MetaEndpoint, or None.

        For non-meta endpoints, returns ``None`` so workbench-bridges renders
        its default (``endpoint_name``'s own count).

        For MetaEndpoints, returns a callable that composes
        :meth:`Endpoint.instance_counts` per async child:
        ``[child_a:2, child_b:1→3]``. Returns an empty string when the meta
        has no async children, which suppresses the ``instances=`` field.
        """
        dag_dict = meta.get("meta_endpoint_dag")
        if not dag_dict:
            return None

        async_children = [name for name, is_async in dag_dict.get("endpoint_async", {}).items() if is_async]
        if not async_children:
            return lambda: ""

        from workbench.api.endpoint import Endpoint

        def fn() -> str:
            parts = []
            for child_name in async_children:
                # Construction fetches fresh; use the cached read helper
                # to avoid a redundant refresh round-trip.
                counts = Endpoint(child_name)._read_instance_counts()
                if not counts:
                    parts.append(f"{child_name}:?")
                    continue
                c, d = counts["current"], counts["desired"]
                val = str(c) if c == d else f"{c}→{d}"
                parts.append(f"{child_name}:{val}")
            return "[" + ", ".join(parts) + "]"

        return fn
