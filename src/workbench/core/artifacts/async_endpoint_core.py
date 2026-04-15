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

Concurrency model
-----------------
Chunks are submitted to SageMaker in parallel via a ``ThreadPoolExecutor``
with up to ``max_in_flight`` outstanding requests. Each worker thread
handles its own S3 upload, ``invoke_async`` call, and output polling.
Results are reassembled in input order so ``id_column`` flows and row
alignment stay correct. This builds real backlog in SageMaker's queue,
which is what drives the auto-scaling policies.
"""

import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from typing import Optional

import pandas as pd
from botocore.config import Config

from workbench.core.artifacts.endpoint_core import EndpointCore

# SageMaker V3 Resource Classes
from sagemaker.core.resources import Endpoint as SagemakerEndpoint

log = logging.getLogger("workbench")

# Polling parameters for async output (exponential backoff, capped).
_POLL_INITIAL_S = 3
_POLL_MAX_S = 30
_POLL_BACKOFF = 1.5

# Default number of concurrent in-flight invocations. Enough to drive
# backlog past the autoscaling target (2/instance) even once scaled out,
# well under SageMaker's ~1000 pending-per-endpoint queue limit.
_DEFAULT_MAX_IN_FLIGHT = 16

# Default rows per invocation. With the 1-hour SageMaker async timeout, 50 rows
# is safe even for per-row work on the order of a minute, and amortizes the
# S3 upload + invoke_async + polling overhead across more useful work.
# Override per-endpoint via workbench_meta["inference_batch_size"].
_DEFAULT_BATCH_SIZE = 50

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

        # S3 paths for async I/O — these mirror the paths configured in
        # AsyncInferenceConfig during deployment.
        base = f"{self.endpoints_s3_path}/{self.name}"
        self.async_output_path = f"{base}/async-output"
        self.async_failure_path = f"{base}/async-failures"
        self.async_input_path = f"{base}/async-input"

    # -----------------------------------------------------------------
    # Override: endpoint_metrics  (async endpoints have different CloudWatch metrics)
    # -----------------------------------------------------------------
    def endpoint_metrics(self):
        """Return async-specific metrics (ModelLatency, CPU, Memory, BacklogSize, errors)."""
        from workbench.utils.endpoint_metrics import EndpointMetrics

        self.log.important("Updating async endpoint metrics...")
        variant = self.endpoint_meta["ProductionVariants"][0]["VariantName"]
        return EndpointMetrics(preset="async").get_metrics(self.name, variant=variant)

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
    # Internal: async invocation machinery
    # -----------------------------------------------------------------
    def _async_batch_invoke(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Split ``eval_df`` into chunks, invoke in parallel, reassemble in input order.

        Reads two knobs from ``workbench_meta()``:
          * ``inference_batch_size``    (default 50): rows per invocation
          * ``inference_max_in_flight`` (default 16): concurrent in-flight requests

        Failed chunks are logged and dropped from the output (we raise only
        if *every* chunk fails — silently returning an empty DataFrame with
        the input columns would be a lie).
        """
        meta = self.workbench_meta() or {}
        batch_size = int(meta.get("inference_batch_size", _DEFAULT_BATCH_SIZE))
        max_in_flight = int(meta.get("inference_max_in_flight", _DEFAULT_MAX_IN_FLIGHT))

        # Size the connection pool to match in-flight concurrency plus headroom.
        # Default botocore pool is 10 — anything larger triggers noisy
        # "Connection pool is full" warnings and forces ad-hoc socket creation.
        client_config = Config(max_pool_connections=max(max_in_flight * 2, 10))
        sm_endpoint = SagemakerEndpoint.get(self.endpoint_name, session=self.boto3_session)
        s3_client = self.boto3_session.client("s3", config=client_config)

        # Slice into (index, chunk) pairs so we can reorder results after the pool returns.
        chunks = [
            (i, eval_df.iloc[start : start + batch_size])
            for i, start in enumerate(range(0, len(eval_df), batch_size))
        ]
        total = len(chunks)
        log.important(
            f"Async batch invoke: {len(eval_df)} rows, batch_size={batch_size}, "
            f"chunks={total}, max_in_flight={max_in_flight}"
        )

        results: dict[int, pd.DataFrame] = {}
        failed_indices: list[int] = []

        with ThreadPoolExecutor(max_workers=max_in_flight) as pool:
            future_to_idx = {
                pool.submit(self._invoke_one_async, sm_endpoint, s3_client, chunk): idx
                for idx, chunk in chunks
            }
            done = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                done += 1
                try:
                    result_df = fut.result()
                except Exception as e:
                    log.error(f"Chunk {idx} raised unexpectedly: {e}")
                    failed_indices.append(idx)
                    continue

                if result_df is None or result_df.empty:
                    failed_indices.append(idx)
                    continue

                results[idx] = result_df
                if done % 25 == 0 or done == total:
                    log.info(f"Async progress: {done}/{total} chunks complete ({len(failed_indices)} failed)")

        if not results:
            raise RuntimeError(
                f"All {total} async invocations failed for endpoint '{self.endpoint_name}'"
            )
        if failed_indices:
            log.warning(
                f"{len(failed_indices)} of {total} chunks failed for '{self.endpoint_name}' "
                f"(indices: {sorted(failed_indices)[:10]}{'...' if len(failed_indices) > 10 else ''})"
            )

        # Reassemble in input order.
        ordered = [results[i] for i in sorted(results)]
        combined_df = pd.concat(ordered, ignore_index=True)
        return self._type_conversions(combined_df)

    def _invoke_one_async(
        self,
        sm_endpoint: SagemakerEndpoint,
        s3_client,
        chunk_df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """Upload one chunk to S3, call invoke_async, poll for output, download result.

        Input is always cleaned up in a ``finally`` block so failures don't
        leak CSV files into S3. Successful output files are cleaned up too.
        """
        request_id = uuid.uuid4().hex
        t_start = time.time()

        # Upload input CSV to S3
        csv_buffer = StringIO()
        chunk_df.to_csv(csv_buffer, index=False)
        input_key = self._s3_key(f"async-input/{request_id}.csv")
        input_s3_uri = f"s3://{self.workbench_bucket}/{input_key}"

        output_location = None
        try:
            s3_client.put_object(
                Bucket=self.workbench_bucket,
                Key=input_key,
                Body=csv_buffer.getvalue(),
                ContentType="text/csv",
            )

            try:
                response = sm_endpoint.invoke_async(
                    input_location=input_s3_uri,
                    content_type="text/csv",
                    accept="text/csv",
                )
                output_location = response.output_location
            except Exception as e:
                log.error(f"invoke_async failed for request {request_id}: {e}")
                return None

            # Poll for the output
            try:
                output_body = self._poll_s3_output(s3_client, output_location)
            except TimeoutError:
                log.error(f"Async request {request_id} timed out waiting for output")
                return None
            except Exception as e:
                log.error(f"Async request {request_id} failed: {e}")
                return None

            # Parse the output CSV
            try:
                result_df = pd.read_csv(StringIO(output_body))
            except Exception as e:
                log.error(f"Failed to parse async output for {request_id}: {e}")
                return None

            elapsed = time.time() - t_start
            log.info(f"Async {request_id[:8]}: {len(result_df)} rows in {elapsed:.1f}s")
            return result_df

        finally:
            # Always clean up the input CSV, even on failure.
            self._cleanup_s3(s3_client, input_s3_uri)
            # Clean up the output file only if we got one (success path).
            if output_location is not None:
                self._cleanup_s3(s3_client, output_location)

    def _poll_s3_output(self, s3_client, output_location: str) -> str:
        """Poll S3 until the output file appears, then download and return its content.

        Uses exponential backoff from ``_POLL_INITIAL_S`` up to ``_POLL_MAX_S``.
        Deadline is 3600s to match SageMaker's max async invocation timeout.
        """
        bucket, key = self._parse_s3_uri(output_location)
        # Failure file is at the same key under the failure prefix
        failure_key = key.replace("/async-output/", "/async-failures/", 1)

        deadline = time.time() + 3600
        interval = _POLL_INITIAL_S

        while time.time() < deadline:
            # Check for success
            try:
                resp = s3_client.get_object(Bucket=bucket, Key=key)
                return resp["Body"].read().decode("utf-8")
            except s3_client.exceptions.NoSuchKey:
                pass

            # Check for failure
            try:
                resp = s3_client.get_object(Bucket=bucket, Key=failure_key)
                failure_body = resp["Body"].read().decode("utf-8")
                raise RuntimeError(f"Async inference failed: {failure_body[:500]}")
            except s3_client.exceptions.NoSuchKey:
                pass

            time.sleep(interval)
            interval = min(interval * _POLL_BACKOFF, _POLL_MAX_S)

        raise TimeoutError(f"Async output not available after 3600s: {output_location}")

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------
    def _cleanup_s3(self, s3_client, *s3_uris: str) -> None:
        """Delete S3 objects after we've consumed them. Best-effort — failures are logged, not raised."""
        for uri in s3_uris:
            try:
                bucket, key = self._parse_s3_uri(uri)
                s3_client.delete_object(Bucket=bucket, Key=key)
            except Exception as e:
                log.debug(f"Failed to clean up {uri}: {e}")

    def _s3_key(self, suffix: str) -> str:
        """Build an S3 key under this endpoint's namespace."""
        return f"endpoints/{self.name}/{suffix}"

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple:
        """Parse 's3://bucket/key' into (bucket, key)."""
        if not uri.startswith("s3://"):
            raise ValueError(f"Not an S3 URI: {uri}")
        without_prefix = uri[5:]
        bucket, _, key = without_prefix.partition("/")
        return bucket, key

    @staticmethod
    def _type_conversions(df: pd.DataFrame) -> pd.DataFrame:
        """Post-process async output: N/A → NaN, type conversions.

        Same logic as EndpointCore._predict's post-processing.
        """
        # N/A → NaN
        na_counts = df.isin(["N/A"]).sum()
        for column, count in na_counts.items():
            if count > 0:
                log.warning(f"{column} has {count} N/A values, converting to NaN")
        df = df.replace("N/A", float("nan"))

        # Hard numeric conversion
        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column])
            except (ValueError, TypeError):
                pass

        # Soft conversion
        df = df.convert_dtypes()
        df.replace("__NA__", pd.NA, inplace=True)

        # Boolean detection
        for column in df.select_dtypes(include=["string"]).columns:
            if df[column].str.lower().isin(["true", "false"]).all():
                df[column] = df[column].str.lower().map({"true": True, "false": False})

        return df
