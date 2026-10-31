"""Low-level helpers for :class:`~workbench.core.artifacts.async_endpoint_core.AsyncEndpointCore`.

Batch-sizing knobs and the MetaEndpoint log-decoration callables live here so
the class stays focused on the inference flow rather than the plumbing.

Imported by ``workbench.core.artifacts.async_endpoint_core``.
"""

import logging
from typing import Optional

from workbench.endpoints.async_inference import resolve_boto_session

log = logging.getLogger("workbench")

# Rows per invocation, per vCPU on the serving box, for an async endpoint that does not choose
# its own batch size at deploy. A batch is the unit of work one server slot takes, and the
# host-side stages pool across the rows inside it (conformer generation, featurization), so a
# batch narrower than the box leaves vCPUs idle. Two rows per vCPU gives that pool a second
# task per worker to even out a straggler. A model forward on an accelerator is chunked inside
# the container instead, so it does not widen with the batch: past filling the vCPUs, batch
# width only amortizes per-invocation overhead. Going wider costs invocation length, which the
# 60-minute timeout bounds.
TASKS_PER_VCPU = 2

# Rows per invocation for an async endpoint with no stamped batch size, or no vCPU count to
# size against at deploy.
ASYNC_BATCH_SIZE = 10

# Outstanding invocations per server-side slot, where a slot is one concurrent
# invocation the endpoint accepts at full scale (see :func:`async_max_in_flight`).
# Two keeps exactly one chunk queued behind each running one, which is what both
# halves of the system need:
#
# - The fleet stays fed: a slot never idles waiting for the client to notice a
#   completion and submit the next chunk.
# - Scale-out has a signal. It fires on ApproximateBacklogSize >= 1, which counts queued
#   work rather than work in flight, so one chunk per slot would hold a busy endpoint at
#   backlog 0 and leave it at whatever instance count it started with.
#
# Queuing deeper than the fleet can drain is what makes a slow endpoint lose work.
# The client's poll deadline starts at submission, so a chunk sitting N deep behind a
# service time of S resolves at N*S and is abandoned once that exceeds the deadline,
# however well the endpoint is working.
IN_FLIGHT_PER_SLOT = 2


class EndpointWarmingError(Exception):
    """An async endpoint (or one of its async children) is still cold-starting
    past the warm-up cap.

    Retryable by design: the cluster is scaling, it just isn't serving yet.
    Raised so callers that treat async endpoints like regular ones get a clear,
    catchable signal to retry shortly rather than a hung or silently-NaN batch.
    """


_SIZE_VCPUS = {"medium": 1, "large": 2, "xlarge": 4}


def instance_vcpus(instance_type: str) -> int:
    """vCPUs on an ``ml.`` instance type, read off its size suffix.

    Modern families all follow one pattern -- ``large`` is 2, ``xlarge`` is 4 and
    ``Nxlarge`` is 4N -- so the name carries the count and there is no table to keep
    current. Returns 0 for a size outside that pattern (``metal``, accelerator sizes),
    which leaves the caller on its own default.
    """
    size = instance_type.rsplit(".", 1)[-1]
    if size in _SIZE_VCPUS:
        return _SIZE_VCPUS[size]
    prefix = size[: -len("xlarge")] if size.endswith("xlarge") else ""
    return 4 * int(prefix) if prefix.isdigit() else 0


def warn_on_fallback_pool(endpoint_name: str, endpoint: dict, config: dict) -> None:
    """Log when a variant serves from anything but its first-choice instance pool.

    The endpoint reports InService from any pool, so this is the one signal that the
    region is short on the type the endpoint asked for. It explains a run slower than the
    preferred pool's throughput, and it is the warning that a pool list which has dropped
    to its last entries is one shortage away from having nowhere to go.
    """
    ordered = sorted(config["ProductionVariants"][0].get("InstancePools") or [], key=lambda p: p["Priority"])
    if len(ordered) < 2:
        return
    serving = {
        pool["InstanceType"]
        for pool in endpoint["ProductionVariants"][0].get("InstancePools") or []
        if pool.get("CurrentInstanceCount")
    }
    preferred = ordered[0]["InstanceType"]
    if serving and preferred not in serving:
        log.warning(
            f"{endpoint_name}: serving from {', '.join(sorted(serving))} rather than its first-choice "
            f"pool {preferred} -- the region is short on {preferred}. Scaling to zero and warming "
            f"again re-attempts it."
        )


def async_max_in_flight(meta: dict) -> int:
    """Outstanding invocations for an async endpoint, from what it stamped at deploy.

    Its slots at full scale, ``max_instances`` times ``async_max_concurrent``, times
    :data:`IN_FLIGHT_PER_SLOT`. The fleet ceiling rather than the instances currently up,
    because the client's queue is the demand signal the autoscaler measures: sizing against
    what is running would hold a cold endpoint at its floor. An endpoint with neither stamp
    reads as one slot, which costs throughput rather than losing chunks.
    """
    return int(meta.get("max_instances", 1)) * int(meta.get("async_max_concurrent", 1)) * IN_FLIGHT_PER_SLOT


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
    callable polls the child queue depth and reports per-child ``<done>/<total>
    chunks`` — a rising ``done`` proves progress, a flat one means stuck.

    The producer stages work in waves (one per meta-batch), so the queue depth
    both rises (new arrivals) and falls (workers finishing) over the run. We
    accumulate the per-poll deltas — arrivals into ``total``, departures into
    ``done`` — so both counters are monotonic and survive across waves, rather
    than resetting each time a new wave pushes the depth past its prior peak.

    Returns ``None`` for non-meta endpoints or metas with no async children.
    """
    children = _async_children(meta)
    if not children:
        return None

    prev_depth: dict[str, int] = {}
    total: dict[str, int] = {}  # cumulative arrivals (chunks ever staged)
    done: dict[str, int] = {}  # cumulative departures (chunks drained)

    def fn() -> str:
        parts = []
        for child_name in children:
            try:
                depth = len(list_async_queue(child_name, s3_bucket, sm_session=sm_session))
            except Exception as e:
                # A progress readout must never break the inference it reports on.
                log.debug(f"progress poll failed for {child_name}: {e}")
                continue
            if child_name in prev_depth:
                delta = depth - prev_depth[child_name]
                total[child_name] = total.get(child_name, 0) + max(0, delta)  # arrivals
                done[child_name] = done.get(child_name, 0) + max(0, -delta)  # departures
            else:
                total[child_name] = depth  # first poll: seed total with the standing queue
                done[child_name] = 0
            prev_depth[child_name] = depth
            if total[child_name] == 0:
                parts.append(f"{child_name}: queued")  # fan-out not started yet
            else:
                parts.append(f"{child_name}: {done[child_name]}/{total[child_name]} chunks")
        return " | ".join(parts)

    return fn
