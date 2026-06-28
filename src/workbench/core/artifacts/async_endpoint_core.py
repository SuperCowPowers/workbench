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
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from workbench.core.artifacts.endpoint_core import EndpointCore
from workbench.endpoints.async_inference import async_inference
from workbench.utils.async_endpoint_utils import (
    EndpointWarmingError,
    _async_children,
    build_meta_instances_str_fn,
    build_meta_progress_str_fn,
    purge_async_queue,
    resolve_batch_sizing,
)

log = logging.getLogger("workbench")

# Cold-start warm-up: how long to wait for scale-from-zero before giving up, and
# how often to re-check the live instance count while waiting. The poll is
# deliberately coarse — scale-from-zero takes minutes, and each poll is a
# DescribeEndpoint call (throttle-prone when several children warm in parallel).
WARM_UP_CAP_S = 900  # 15 min — async fleet cold start plus headroom
WARM_UP_POLL_S = 20

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
    # Cold-start warm-up
    # -----------------------------------------------------------------
    def _ensure_warm(self) -> None:
        """Block until this endpoint and all its async children are serving an instance.

        Reactive: a no-op once at least one instance is up (the common warm case,
        and always true when ``min_instances >= 1``). For a MetaEndpoint, the meta's
        own self-warm runs *in parallel* with its children rather than after them —
        scale-from-zero of the meta instance is independent of the children, so
        warming every level at once roughly halves the cascading cold-start wall
        clock. Correctness is preserved by joining on *all* warmers before returning:
        the real batch only fires once every level is confirmed warm. (A meta warmer
        fired while a child is still cold may churn or time out cascading into the
        cold child, but that's harmless — readiness is judged by instance count, not
        by the throwaway warmer's result.)

        Raises :class:`EndpointWarmingError` if warm-up exceeds ``WARM_UP_CAP_S``.
        """
        children = _async_children(self.workbench_meta() or {})
        warmers = [AsyncEndpointCore(c)._ensure_warm for c in children]
        warmers.append(self._warm_self)  # always >= 1 warmer, so max_workers >= 1
        with ThreadPoolExecutor(max_workers=len(warmers)) as pool:
            futures = [pool.submit(w) for w in warmers]
            for fut in futures:
                fut.result()  # propagate any EndpointWarmingError

    def _warm_self(self) -> None:
        """Fire a warm-up request, then poll until an instance is serving.

        Always fires the warmer (one trivial async job) rather than checking first:
        simpler, and it covers fresh deploys where a transient instance would no-op a
        reactive check and then decay — the in-flight warmer keeps the endpoint alive.
        Serverless short-circuits (no instance count to poll).
        """
        if self.is_serverless():
            return  # serverless scales per-request — no instances to warm, no count to poll

        log.important(f"Endpoint '{self.name}': warm-up request (cold scale-up can take 10-15 minutes)...")

        # Queue a trivial job so SageMaker scales the fleet up. If we can't even
        # queue it (perms, bad payload, missing bucket), that's a hard, non-retryable
        # error — fail fast with the real cause instead of polling a fleet that will
        # never scale for the full cap and then reporting a misleading "retry shortly".
        fire_error = self._fire_warmer()
        if fire_error is not None:
            raise EndpointWarmingError(
                f"Endpoint '{self.name}' could not be warmed — failed to queue a warm-up request: {fire_error}"
            )

        # Check before sleeping so an already-warm instance returns immediately.
        deadline = time.time() + WARM_UP_CAP_S
        while time.time() < deadline:
            if self._current_instances() >= 1:
                log.important(f"Endpoint '{self.name}' is warm.")
                return
            time.sleep(WARM_UP_POLL_S)

        counts = self._live_instance_counts()
        raise EndpointWarmingError(
            f"Endpoint '{self.name}' still warming after {WARM_UP_CAP_S // 60}m "
            f"(instances {counts.get('current', '?')}/{counts.get('desired', '?')} up) — retry shortly."
        )

    def _live_instance_counts(self) -> dict:
        """Fresh ``{'current', 'desired'}`` instance counts (refreshes meta first)."""
        self.refresh_meta()
        return self._read_instance_counts()

    def _current_instances(self) -> int:
        return self._live_instance_counts().get("current", 0)

    def _fire_warmer(self) -> "str | None":
        """Queue one trivial async invocation to trigger scale-from-zero.

        Non-blocking: stages a 1-row input and calls ``invoke_endpoint_async``,
        ignoring the result. Its only purpose is to create queue backlog so
        SageMaker scales the fleet; readiness is judged by the live instance count,
        never by this call's output (which would itself block behind the cold start
        we're waiting on). Returns ``None`` on success, or an error string if the
        request could not be queued (so the caller can fail fast).
        """
        try:
            warmer_csv = self._warmer_df().to_csv(index=False)
            # Stage under a dedicated 'async-warmup' prefix (NOT 'async-input') so the
            # fixed warmer file never counts toward the queue-depth that the meta
            # progress reporter reads from the async-input prefix.
            key = f"endpoints/{self.name}/async-warmup/_warmup.csv"
            self.boto3_session.client("s3").put_object(
                Bucket=self.workbench_bucket, Key=key, Body=warmer_csv, ContentType="text/csv"
            )
            self.boto3_session.client("sagemaker-runtime").invoke_endpoint_async(
                EndpointName=self.endpoint_name,
                InputLocation=f"s3://{self.workbench_bucket}/{key}",
                ContentType="text/csv",
                Accept="text/csv",
            )
            return None
        except Exception as e:
            log.warning(f"Warmer invoke for '{self.name}' failed: {e}")
            return str(e)

    def _warmer_df(self) -> pd.DataFrame:
        """One trivial row to exercise the endpoint — just enough to queue a job.

        Uses the endpoint's declared input columns when available (``smiles`` →
        a real molecule so feature endpoints don't error; other columns → 0),
        falling back to a lone ``smiles`` column. Reads the columns via the cached
        lookup on ``self`` (no fresh ``Endpoint`` construction on the cold path).
        """
        try:
            from workbench.utils.endpoint_utils import (
                input_columns_key,
                lookup_cached_columns,
                register_input_columns,
            )

            cols = lookup_cached_columns(self, input_columns_key(self.name), register_input_columns, "input columns")
            cols = cols or ["smiles"]
        except Exception:
            cols = ["smiles"]
        row = {c: ("CCO" if "smiles" in c.lower() else 0) for c in cols}
        return pd.DataFrame([row])

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
        self._ensure_warm()

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
