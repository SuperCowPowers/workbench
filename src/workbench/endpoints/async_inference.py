"""Async Inference on SageMaker Endpoints.

Lightweight client for invoking SageMaker async endpoints — the long-running
counterpart to ``fast_inference``. Uses the standard async pattern:

    1. Upload chunk CSV → S3
    2. ``invoke_endpoint_async`` → SageMaker returns ``OutputLocation``
    3. Poll the ``OutputLocation`` until the result CSV appears
    4. Download, parse, repeat for the next chunk

Chunks are processed in parallel via a thread pool. Failures are logged and
the failing chunk is dropped from the output (the function only raises if
*every* chunk fails).

This module owns the protocol-level invocation. Workbench's
:class:`AsyncEndpointCore` wraps it with framework-specific concerns
(``workbench_meta`` knobs, capture, monitoring).
"""

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from typing import Callable, Optional

import boto3
import pandas as pd
from botocore.config import Config

from workbench.endpoints.fast_inference import df_type_conversions

log = logging.getLogger("workbench")

# Polling parameters for async output (exponential backoff, capped).
# _POLL_MAX_S is capped at 10s so the per-batch "waiting for the next poll
# after compute actually finished" overhead stays small.
_POLL_INITIAL_S = 3
_POLL_MAX_S = 10
_POLL_BACKOFF = 1.5

# SageMaker async endpoints support up to 60-minute invocations.
_POLL_DEADLINE_S = 3600


def async_inference(
    endpoint_name: str,
    eval_df: pd.DataFrame,
    sm_session=None,
    batch_size: int = 10,
    max_in_flight: int = 64,
    s3_bucket: Optional[str] = None,
    s3_input_prefix: Optional[str] = None,
    instances_str_fn: Optional[Callable[[], str]] = None,
) -> pd.DataFrame:
    """Run async inference on a SageMaker endpoint and return a DataFrame.

    Splits the input into chunks, uploads each to S3 in parallel, calls
    ``invoke_endpoint_async`` for each, polls the corresponding
    ``OutputLocation`` until the result appears, and reassembles the
    chunks in input order.

    Args:
        endpoint_name: Name of the deployed SageMaker async endpoint.
        eval_df: Input DataFrame.
        sm_session: A boto3 ``Session`` (or legacy SageMaker Session with
            ``.boto_session``). If None, a session is created from the
            ambient AWS environment.
        batch_size: Rows per invocation. Smaller batches give better load
            balancing across in-flight workers; larger batches reduce
            per-chunk overhead.
        max_in_flight: Cap on the number of outstanding invocations.
        s3_bucket: S3 bucket for staging input CSVs. **Required.** Must
            be readable by the endpoint's execution role.
        s3_input_prefix: S3 key prefix for staged inputs (defaults to
            ``endpoints/<endpoint_name>/async-input``).
        instances_str_fn: Optional callable returning the string for the
            ``instances=`` log field. Called once at startup and once per
            progress log. When ``None`` (default), the field shows
            ``endpoint_name``'s own current count via :func:`instance_count_str`.
            An empty string suppresses the field. Useful for callers whose
            endpoint count carries no useful signal (e.g. orchestrators
            locked to a single instance).

    Returns:
        DataFrame containing the endpoint's response, with rows in input
        order. Failed chunks are dropped (function raises only if every
        chunk fails).
    """
    if eval_df.empty:
        return pd.DataFrame(columns=eval_df.columns)
    if not s3_bucket:
        raise ValueError("async_inference: s3_bucket is required (where to stage input CSVs)")

    if s3_input_prefix is None:
        s3_input_prefix = f"endpoints/{endpoint_name}/async-input"
    s3_input_prefix = s3_input_prefix.rstrip("/")

    # Build clients with a connection pool sized to the in-flight concurrency.
    # Default botocore pool is 10 — anything larger triggers "Connection pool is
    # full" warnings and forces ad-hoc socket creation.
    boto_session = _resolve_boto_session(sm_session)
    client_config = Config(max_pool_connections=max(max_in_flight * 2, 10))
    s3_client = boto_session.client("s3", config=client_config)
    runtime_client = boto_session.client("sagemaker-runtime", config=client_config)

    # Slice into (index, chunk) pairs so we can reorder results after the pool returns.
    chunks = [
        (i, eval_df.iloc[start : start + batch_size])  # noqa: E203
        for i, start in enumerate(range(0, len(eval_df), batch_size))
    ]
    total = len(chunks)
    actual_in_flight = min(total, max_in_flight)
    sm_client = boto_session.client("sagemaker")

    def _label() -> str:
        return instances_str_fn() if instances_str_fn is not None else instance_count_str(sm_client, endpoint_name)

    instances_label = _label()
    fields = [
        f"endpoint={endpoint_name}",
        f"rows={len(eval_df)}",
        f"batch_size={batch_size}",
        f"chunks={total}",
        f"in_flight={actual_in_flight}",
    ]
    if instances_label:
        fields.append(f"instances={instances_label}")
    log.info("async_inference: " + ", ".join(fields))

    results: dict[int, pd.DataFrame] = {}
    failed_indices: list[int] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=actual_in_flight) as pool:
        futures = {
            pool.submit(
                _invoke_one_async,
                runtime_client,
                s3_client,
                endpoint_name,
                chunk_df,
                s3_bucket,
                s3_input_prefix,
            ): idx
            for idx, chunk_df in chunks
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result_df = fut.result()
            except Exception as e:
                log.error(f"Chunk {idx} raised unexpectedly: {e}")
                failed_indices.append(idx)
                result_df = None
            else:
                if result_df is None or result_df.empty:
                    failed_indices.append(idx)
                else:
                    results[idx] = result_df

            completed += 1
            if completed % 25 == 0 or completed == total:
                progress_instances = _label()
                progress_msg = f"Async progress: {completed}/{total} chunks complete " f"({len(failed_indices)} failed"
                if progress_instances:
                    progress_msg += f", instances={progress_instances}"
                progress_msg += ")"
                log.info(progress_msg)

    if not results:
        raise RuntimeError(f"All {total} async invocations failed for endpoint '{endpoint_name}'")
    if failed_indices:
        log.warning(
            f"{len(failed_indices)} of {total} chunks failed for '{endpoint_name}' "
            f"(indices: {sorted(failed_indices)[:10]}{'...' if len(failed_indices) > 10 else ''})"
        )

    ordered = [results[i] for i in sorted(results)]
    combined_df = pd.concat(ordered, ignore_index=True)
    return df_type_conversions(combined_df)


def instance_count_str(sm_client, endpoint_name: str) -> str:
    """Format an endpoint's current instance count, showing ``current→desired`` while scaling.

    Returns ``"?"`` if the ``describe_endpoint`` call fails (e.g., missing
    IAM permission).
    """
    try:
        variant = sm_client.describe_endpoint(EndpointName=endpoint_name)["ProductionVariants"][0]
        current, desired = variant["CurrentInstanceCount"], variant["DesiredInstanceCount"]
        return str(current) if current == desired else f"{current}→{desired}"
    except Exception:
        return "?"


def _resolve_boto_session(sm_session) -> boto3.Session:
    """Accept a plain boto3.Session, a legacy SageMaker Session (with
    ``.boto_session``), or None (build from ambient env)."""
    if sm_session is not None:
        return getattr(sm_session, "boto_session", sm_session)

    region = (
        os.environ.get("SAGEMAKER_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or boto3.session.Session().region_name
    )
    if not region:
        raise RuntimeError(
            "async_inference: no AWS region configured. Pass sm_session= or set " "SAGEMAKER_REGION / AWS_REGION."
        )
    return boto3.Session(region_name=region)


def _invoke_one_async(
    runtime_client,
    s3_client,
    endpoint_name: str,
    chunk_df: pd.DataFrame,
    s3_bucket: str,
    s3_input_prefix: str,
) -> Optional[pd.DataFrame]:
    """Upload one chunk, invoke async, poll for output, download result.

    Cleans up both input and output S3 objects in a finally block so we
    don't leak CSVs on failure.
    """
    request_id = uuid.uuid4().hex
    t_start = time.time()

    csv_buffer = StringIO()
    chunk_df.to_csv(csv_buffer, index=False)
    input_key = f"{s3_input_prefix}/{request_id}.csv"
    input_s3_uri = f"s3://{s3_bucket}/{input_key}"

    output_location: Optional[str] = None
    try:
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=input_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv",
        )

        try:
            response = runtime_client.invoke_endpoint_async(
                EndpointName=endpoint_name,
                InputLocation=input_s3_uri,
                ContentType="text/csv",
                Accept="text/csv",
            )
            output_location = response["OutputLocation"]
        except Exception as e:
            log.error(f"invoke_endpoint_async failed for request {request_id}: {e}")
            return None

        try:
            output_body = _poll_s3_output(s3_client, output_location)
        except TimeoutError:
            log.error(f"Async request {request_id} timed out waiting for output")
            return None
        except Exception as e:
            log.error(f"Async request {request_id} failed: {e}")
            return None

        try:
            result_df = pd.read_csv(StringIO(output_body))
        except Exception as e:
            log.error(f"Failed to parse async output for {request_id}: {e}")
            return None

        elapsed = time.time() - t_start
        log.info(f"Async {request_id[:8]}: {len(result_df)} rows in {elapsed:.1f}s")
        return result_df

    finally:
        _cleanup_s3(s3_client, input_s3_uri)
        if output_location is not None:
            _cleanup_s3(s3_client, output_location)


def _poll_s3_output(s3_client, output_location: str) -> str:
    """Poll S3 until the output file appears, then download and return its content.

    Uses exponential backoff from ``_POLL_INITIAL_S`` up to ``_POLL_MAX_S``.
    The deadline matches SageMaker's max async invocation timeout (60 minutes).
    """
    bucket, key = _parse_s3_uri(output_location)
    failure_key = key.replace("/async-output/", "/async-failures/", 1)

    deadline = time.time() + _POLL_DEADLINE_S
    interval = _POLL_INITIAL_S

    while time.time() < deadline:
        try:
            resp = s3_client.get_object(Bucket=bucket, Key=key)
            return resp["Body"].read().decode("utf-8")
        except s3_client.exceptions.NoSuchKey:
            pass

        try:
            resp = s3_client.get_object(Bucket=bucket, Key=failure_key)
            failure_body = resp["Body"].read().decode("utf-8")
            raise RuntimeError(f"Async inference failed: {failure_body[:500]}")
        except s3_client.exceptions.NoSuchKey:
            pass

        time.sleep(interval)
        interval = min(interval * _POLL_BACKOFF, _POLL_MAX_S)

    raise TimeoutError(f"Async output not available after {_POLL_DEADLINE_S}s: {output_location}")


def purge_async_queue(
    endpoint_name: str,
    s3_bucket: str,
    sm_session=None,
    s3_input_prefix: Optional[str] = None,
) -> int:
    """Cancel all queued async invocations for an endpoint.

    SageMaker async invocations reference an input CSV in S3. Deleting those
    staged inputs causes any not-yet-pulled invocation to fail fast with
    ``NoSuchKey`` when SageMaker tries to read it — draining the queue
    without further compute. In-flight invocations are unaffected (they've
    already pulled their input).

    Run this only when no live callers are dispatching against the endpoint;
    otherwise their just-uploaded inputs may be deleted before SageMaker
    pulls them.

    Args:
        endpoint_name: Name of the deployed SageMaker async endpoint.
        s3_bucket: Bucket where async-input CSVs are staged.
        sm_session: Optional boto3/SageMaker session (see
            :func:`async_inference`). If None, builds one from the ambient
            AWS environment.
        s3_input_prefix: S3 key prefix for staged inputs. Defaults to
            ``endpoints/<endpoint_name>/async-input`` to match
            :func:`async_inference`'s upload location.

    Returns:
        int: Number of staged input objects deleted.
    """
    if s3_input_prefix is None:
        s3_input_prefix = f"endpoints/{endpoint_name}/async-input"
    prefix = s3_input_prefix.rstrip("/") + "/"

    boto_session = _resolve_boto_session(sm_session)
    s3_client = boto_session.client("s3")

    deleted = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        objs = page.get("Contents") or []
        if not objs:
            continue
        keys = [{"Key": o["Key"]} for o in objs]
        s3_client.delete_objects(Bucket=s3_bucket, Delete={"Objects": keys, "Quiet": True})
        deleted += len(keys)

    log.info(f"purge_async_queue: deleted {deleted} staged inputs from s3://{s3_bucket}/{prefix}")
    return deleted


def _cleanup_s3(s3_client, *s3_uris: str) -> None:
    """Best-effort cleanup of S3 objects. Failures are logged, not raised."""
    for uri in s3_uris:
        try:
            bucket, key = _parse_s3_uri(uri)
            s3_client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            log.debug(f"Failed to clean up {uri}: {e}")


def _parse_s3_uri(uri: str) -> tuple:
    """Parse 's3://bucket/key' into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    without_prefix = uri[5:]
    bucket, _, key = without_prefix.partition("/")
    return bucket, key
