"""Manual verification harness for async endpoint auto-scaling.

Submits a large inference call at a named async endpoint and prints a timeline
of ``DesiredInstanceCount``, ``HasBacklogWithoutCapacity`` and
``ApproximateBacklogSizePerInstance`` so you can *see* the scale-out happen.

Usage:
    python scripts/admin/verify_async_autoscaling.py <endpoint_name> [--chunks 32] [--minutes 15]

The load is a single ``endpoint.inference()`` call with ``chunks × batch_size``
unique SMILES pulled from ``PublicData().get("comp_chem/aqsol/aqsol_public_data")``.
``AsyncEndpointCore`` splits the frame into chunks and submits them in parallel
via its internal thread pool (up to ``inference_max_in_flight`` concurrent).
Assumes the endpoint accepts a ``smiles`` column.

Expected behavior on a correctly-configured scale-to-zero async endpoint,
starting from 0 instances:

    t=0s    instances=0  has_backlog=1  backlog_per=--
    t=60s   instances=0  has_backlog=1  backlog_per=--
    t=120s  instances=1  has_backlog=0  backlog_per=large   ← step policy fired (0→1)
    t=180s  instances=1  has_backlog=0  backlog_per=large
    t=300s  instances=3  has_backlog=0  backlog_per=~2      ← target tracking added instances
    ...

If ``instances`` stays at 0 or 1 while ``has_backlog`` is consistently 1, the
step-scaling policy isn't wired up. Check:

    aws application-autoscaling describe-scaling-policies \\
        --service-namespace sagemaker \\
        --resource-id endpoint/<name>/variant/AllTraffic
"""

import argparse
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

from workbench.api import AsyncEndpoint, PublicData


def _get_metric(cw, endpoint_name: str, metric: str, stat: str, start, end) -> float:
    """Single-point read of a CloudWatch metric for this endpoint."""
    resp = cw.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName=metric,
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        StartTime=start,
        EndTime=end,
        Period=60,
        Statistics=[stat],
    )
    pts = sorted(resp.get("Datapoints", []), key=lambda p: p["Timestamp"])
    return pts[-1][stat] if pts else None


def _describe_capacity(sm, endpoint_name: str) -> int:
    """Current DesiredInstanceCount for the single production variant."""
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        return desc["ProductionVariants"][0].get("DesiredInstanceCount", 0)
    except Exception as e:
        print(f"  [describe_endpoint failed: {e}]")
        return -1


def monitor_timeline(endpoint_name: str, boto3_session, minutes: int, worker=None) -> None:
    """Poll every 30s for ``minutes`` and print a timeline.

    Exits early when ``worker`` (the inference thread) finishes, so the monitor
    tracks the full load cycle but doesn't hang past it.
    """
    cw = boto3_session.client("cloudwatch")
    sm = boto3_session.client("sagemaker")

    print(f"\n{'t':>6s}  {'instances':>9s}  {'has_backlog':>11s}  {'backlog_per_instance':>21s}")
    print("-" * 56)

    t_start = time.time()
    deadline = t_start + minutes * 60
    while time.time() < deadline:
        t_rel = int(time.time() - t_start)
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=3)

        instances = _describe_capacity(sm, endpoint_name)
        has_backlog = _get_metric(cw, endpoint_name, "HasBacklogWithoutCapacity", "Maximum", start, end)
        per_instance = _get_metric(cw, endpoint_name, "ApproximateBacklogSizePerInstance", "Average", start, end)

        hb = "--" if has_backlog is None else f"{has_backlog:.0f}"
        bp = "--" if per_instance is None else f"{per_instance:.1f}"
        print(f"{t_rel:>5d}s  {instances:>9d}  {hb:>11s}  {bp:>21s}")

        # Once inference completes, keep monitoring a short while to capture
        # the scale-in, then exit instead of waiting out the full deadline.
        if worker is not None and not worker.is_alive():
            if not hasattr(monitor_timeline, "_finished_at"):
                monitor_timeline._finished_at = time.time()
                print(f"  [inference thread done — monitoring scale-in for 5 more minutes]")
            elif time.time() - monitor_timeline._finished_at > 300:
                print(f"  [scale-in window elapsed, exiting]")
                return

        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("endpoint_name", help="Name of the async endpoint to exercise")
    parser.add_argument(
        "--chunks",
        type=int,
        default=32,
        help="Number of chunks to submit (default 32). " "Total SMILES = chunks × endpoint's inference_batch_size.",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=60,
        help="Max time to monitor (default 60 min). Monitor exits early when inference finishes.",
    )
    args = parser.parse_args()

    endpoint = AsyncEndpoint(args.endpoint_name)
    if not endpoint.exists():
        print(f"Endpoint '{args.endpoint_name}' not found.", file=sys.stderr)
        sys.exit(1)

    # Size the load to the endpoint's configured batch size so chunk count matches --chunks.
    meta = endpoint.workbench_meta() or {}
    batch_size = int(meta.get("inference_batch_size", 50))
    needed = args.chunks * batch_size

    aqsol = PublicData().get("comp_chem/aqsol/aqsol_public_data")
    aqsol.columns = aqsol.columns.str.lower()
    if len(aqsol) < needed:
        print(
            f"aqsol has only {len(aqsol)} rows, need {needed} for chunks={args.chunks} × batch_size={batch_size}",
            file=sys.stderr,
        )
        sys.exit(1)
    smiles_df = aqsol[["smiles"]].iloc[:needed].reset_index(drop=True)

    max_in_flight = int(meta.get("inference_max_in_flight", 16))
    print(
        f"Firing one inference() at '{args.endpoint_name}': "
        f"{needed} SMILES → {args.chunks} chunks of {batch_size}, "
        f"up to {max_in_flight} in-flight at a time..."
    )

    # Fire inference in a background thread so the monitor can run in the main thread.
    # A single inference() call reuses internal state (ModelCore, clients) across
    # all chunks — avoids the ListTags throttling you'd see with N parallel callers.
    bg = threading.Thread(
        target=lambda: endpoint.inference(smiles_df),
        daemon=True,
        name="verify-async-load",
    )
    bg.start()

    try:
        monitor_timeline(args.endpoint_name, endpoint.boto3_session, args.minutes, worker=bg)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print(f"\nInference thread alive: {bg.is_alive()}")


if __name__ == "__main__":
    main()
