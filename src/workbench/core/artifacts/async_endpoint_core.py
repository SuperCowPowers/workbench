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
from workbench.endpoints.async_inference import async_inference
from workbench.utils.async_endpoint_utils import (
    build_meta_instances_str_fn,
    build_meta_progress_str_fn,
    purge_async_queue,
    resolve_batch_sizing,
)

log = logging.getLogger("workbench")

# Pandas option applied once at import — avoid mutating global state per call.
pd.set_option("future.no_silent_downcasting", True)


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

        Thin wrapper over :func:`workbench.utils.async_endpoint_utils.purge_async_queue`.
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

        Sizing (``batch_size``/``max_in_flight``) comes from
        :func:`~workbench.utils.async_endpoint_utils.resolve_batch_sizing` via
        the ``inference_batch_size`` / ``inference_max_in_flight`` meta knobs.

        For MetaEndpoints, two log-decoration callables are passed so the
        otherwise-opaque single meta invocation reports useful detail:
        ``instances_str_fn`` (per-child instance counts in the ``instances=``
        field) and ``progress_str_fn`` (per-child queue drain, surfaced in the
        heartbeat). Both are ``None`` for non-meta endpoints.
        """
        meta = self.workbench_meta() or {}
        batch_size, max_in_flight = resolve_batch_sizing(meta, len(eval_df))

        return async_inference(
            endpoint_name=self.endpoint_name,
            eval_df=eval_df,
            sm_session=self.boto3_session,
            batch_size=batch_size,
            max_in_flight=max_in_flight,
            s3_bucket=self.workbench_bucket,
            s3_input_prefix=f"endpoints/{self.name}/async-input",
            instances_str_fn=build_meta_instances_str_fn(meta),
            progress_str_fn=build_meta_progress_str_fn(meta, self.boto3_session, self.workbench_bucket),
        )
