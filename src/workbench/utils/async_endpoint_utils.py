"""Low-level helpers for :class:`~workbench.core.artifacts.async_endpoint_core.AsyncEndpointCore`.

Batch-sizing knobs and the MetaEndpoint log-decoration callables live here so
the class stays focused on the inference flow rather than the plumbing.

Imported by ``workbench.core.artifacts.async_endpoint_core``.
"""

import logging
from typing import Optional

from workbench.endpoints.async_inference import resolve_boto_session

log = logging.getLogger("workbench")

# Default rows per invocation. Smaller batches give better load balancing across
# workers — a handful of slow rows in one chunk stretches total time less when
# there are more chunks to absorb the variance. At ~20s/row (typical async
# workload) the extra per-chunk overhead (~3s polling startup) is <2%. Fast
# endpoints (sub-second per row) should override higher via meta so the overhead
# doesn't dominate. Override via workbench_meta["inference_batch_size"].
DEFAULT_BATCH_SIZE = 10

# Safety cap on client-side thread-pool size for direct (non-InferenceCache)
# calls with large DataFrames. Prevents thread-pool blowup on calls like
# ``end.inference(huge_df)``. Override via workbench_meta["inference_max_in_flight"].
MAX_IN_FLIGHT_CAP = 64


def resolve_batch_sizing(meta: dict, n_rows: int) -> tuple[int, int]:
    """Resolve ``(batch_size, max_in_flight)`` for one async batch invoke.

    ``batch_size`` comes from ``inference_batch_size`` (default
    :data:`DEFAULT_BATCH_SIZE`). ``max_in_flight`` defaults to the resulting
    chunk count (one client-side worker per chunk, fully parallel) but is capped
    by ``inference_max_in_flight`` (default :data:`MAX_IN_FLIGHT_CAP`) to avoid
    thread-pool blowup on very large frames.
    """
    batch_size = int(meta.get("inference_batch_size", DEFAULT_BATCH_SIZE))
    n_batches = max(1, (n_rows + batch_size - 1) // batch_size)
    cap = int(meta.get("inference_max_in_flight", MAX_IN_FLIGHT_CAP))
    return batch_size, min(n_batches, cap)


# ---------------------------------------------------------------------------
# Async queue management (list + purge of staged S3 inputs)
# ---------------------------------------------------------------------------
def list_async_queue(
    endpoint_name: str,
    s3_bucket: str,
    sm_session=None,
    s3_input_prefix: Optional[str] = None,
) -> list[dict]:
    """List staged async-input objects (outstanding + orphaned work) for an endpoint.

    Each entry is a chunk CSV uploaded for an async invocation but not yet
    cleaned up. SageMaker never deletes inputs and the client only removes them
    after collecting output, so this *approximates* outstanding work — queued
    chunks, in-flight chunks, and orphans left by killed callers — rather than a
    precise live queue depth. For the authoritative pending count, use
    SageMaker's ``ApproximateBacklogSize`` CloudWatch metric.

    Args:
        endpoint_name: Name of the deployed SageMaker async endpoint.
        s3_bucket: Bucket where async-input CSVs are staged.
        sm_session: Optional boto3/SageMaker session. If None, builds one from
            the ambient AWS environment.
        s3_input_prefix: S3 key prefix for staged inputs. Defaults to
            ``endpoints/<endpoint_name>/async-input`` to match
            :func:`~workbench.endpoints.async_inference.async_inference`'s
            upload location.

    Returns:
        list[dict]: One entry per staged input, oldest first, each with
        ``endpoint`` (str), ``key`` (str), ``last_modified`` (datetime),
        and ``size`` (int).
    """
    if s3_input_prefix is None:
        s3_input_prefix = f"endpoints/{endpoint_name}/async-input"
    prefix = s3_input_prefix.rstrip("/") + "/"

    boto_session = resolve_boto_session(sm_session)
    s3_client = boto_session.client("s3")

    items: list[dict] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            items.append(
                {
                    "endpoint": endpoint_name,
                    "key": obj["Key"],
                    "last_modified": obj["LastModified"],
                    "size": obj["Size"],
                }
            )

    items.sort(key=lambda i: i["last_modified"])
    return items


def purge_async_queue(
    endpoint_name: str,
    s3_bucket: str,
    sm_session=None,
    s3_input_prefix: Optional[str] = None,
) -> int:
    """Cancel all queued async invocations for an endpoint.

    SageMaker async invocations reference an input CSV in S3. Deleting those
    staged inputs causes any not-yet-pulled invocation to fail fast with
    ``NoSuchKey`` when SageMaker tries to read it — draining the queue without
    further compute. In-flight invocations are unaffected (they've already
    pulled their input).

    Run this only when no live callers are dispatching against the endpoint;
    otherwise their just-uploaded inputs may be deleted before SageMaker pulls
    them.

    Args:
        endpoint_name: Name of the deployed SageMaker async endpoint.
        s3_bucket: Bucket where async-input CSVs are staged.
        sm_session: Optional boto3/SageMaker session. If None, builds one from
            the ambient AWS environment.
        s3_input_prefix: S3 key prefix for staged inputs. Defaults to
            ``endpoints/<endpoint_name>/async-input``.

    Returns:
        int: Number of staged input objects deleted.
    """
    items = list_async_queue(endpoint_name, s3_bucket, sm_session=sm_session, s3_input_prefix=s3_input_prefix)
    if not items:
        log.info(f"purge_async_queue: no staged inputs for endpoint '{endpoint_name}'")
        return 0

    boto_session = resolve_boto_session(sm_session)
    s3_client = boto_session.client("s3")

    # delete_objects accepts up to 1000 keys per request.
    deleted = 0
    for start in range(0, len(items), 1000):
        batch = [{"Key": i["key"]} for i in items[start : start + 1000]]  # noqa: E203
        s3_client.delete_objects(Bucket=s3_bucket, Delete={"Objects": batch, "Quiet": True})
        deleted += len(batch)

    log.info(f"purge_async_queue: deleted {deleted} staged inputs for endpoint '{endpoint_name}'")
    return deleted


def _async_children(meta: dict) -> list[str]:
    """Async child endpoint names from a MetaEndpoint's serialized DAG (may be empty)."""
    dag_dict = meta.get("meta_endpoint_dag")
    if not dag_dict:
        return []
    return [name for name, is_async in dag_dict.get("endpoint_async", {}).items() if is_async]


def build_meta_instances_str_fn(meta: dict):
    """Build the ``instances_str_fn`` for async_inference's ``instances=`` log field.

    Returns ``None`` for non-meta endpoints (async_inference renders its own
    endpoint's count). For a MetaEndpoint, returns a callable rendering each
    async child's current instance count — ``[child_a:2, child_b:1→3]`` — or a
    lambda returning ``""`` when there are no async children (suppresses the field).
    """
    if not meta.get("meta_endpoint_dag"):
        return None
    children = _async_children(meta)
    if not children:
        return lambda: ""

    from workbench.api.endpoint import Endpoint

    def fn() -> str:
        parts = []
        for child_name in children:
            # Cached read helper avoids a redundant refresh round-trip.
            counts = Endpoint(child_name)._read_instance_counts()
            if not counts:
                parts.append(f"{child_name}:?")
                continue
            c, d = counts["current"], counts["desired"]
            val = str(c) if c == d else f"{c}→{d}"
            parts.append(f"{child_name}:{val}")
        return "[" + ", ".join(parts) + "]"

    return fn


def build_meta_progress_str_fn(meta: dict, sm_session, s3_bucket: str):
    """Build the heartbeat ``progress_str_fn`` for a MetaEndpoint, or ``None``.

    A meta invocation is opaque to the client (one chunk), but each async child
    drains its own staged-input queue as the server-side fan-out runs. This
    callable polls the child queue depth and reports per-child ``<done>/<peak>
    chunks`` — a rising ``done`` proves progress, a flat one means stuck. ``peak``
    is the largest depth seen so far, which approximates the child's total chunk
    count once the fan-out is fully staged.

    Returns ``None`` for non-meta endpoints or metas with no async children.
    """
    children = _async_children(meta)
    if not children:
        return None

    peak: dict[str, int] = {}

    def fn() -> str:
        parts = []
        for child_name in children:
            try:
                depth = len(list_async_queue(child_name, s3_bucket, sm_session=sm_session))
            except Exception as e:
                # A progress readout must never break the inference it reports on.
                log.debug(f"progress poll failed for {child_name}: {e}")
                continue
            peak[child_name] = max(peak.get(child_name, 0), depth)
            if peak[child_name] == 0:
                parts.append(f"{child_name}: queued")  # fan-out not started yet
            else:
                parts.append(f"{child_name}: {peak[child_name] - depth}/{peak[child_name]} chunks")
        return " | ".join(parts)

    return fn
