"""Manual verification harness for async endpoint auto-scaling.

Fires N parallel invocations at a named async endpoint and prints a timeline
of ``DesiredInstanceCount``, ``HasBacklogWithoutCapacity`` and
``ApproximateBacklogSizePerInstance`` so you can *see* the scale-out happen.

Usage:
    python scripts/verify_async_autoscaling.py <endpoint_name> <sample_csv> [--n 32] [--minutes 15]

``sample_csv`` should be a small CSV (1-10 rows) with the columns the endpoint
expects — it's cloned N times to generate the load.

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
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from workbench.api import AsyncEndpoint


def _fire_one(endpoint: AsyncEndpoint, sample_df: pd.DataFrame) -> int:
    """Submit one inference request. Returns the elapsed seconds."""
    t0 = time.time()
    _ = endpoint.inference(sample_df)
    return int(time.time() - t0)


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


def monitor_timeline(endpoint_name: str, boto3_session, minutes: int) -> None:
    """Poll every 30s for ``minutes`` and print a timeline."""
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
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("endpoint_name", help="Name of the async endpoint to exercise")
    parser.add_argument("sample_csv", help="Small CSV with the columns the endpoint expects")
    parser.add_argument("--n", type=int, default=32, help="Number of parallel invocations (default 32)")
    parser.add_argument("--minutes", type=int, default=15, help="How long to monitor (default 15 min)")
    args = parser.parse_args()

    endpoint = AsyncEndpoint(args.endpoint_name)
    if not endpoint.exists():
        print(f"Endpoint '{args.endpoint_name}' not found.", file=sys.stderr)
        sys.exit(1)

    sample_df = pd.read_csv(args.sample_csv)
    print(f"Firing {args.n} parallel inference requests at '{args.endpoint_name}' "
          f"(sample: {len(sample_df)} rows × {len(sample_df.columns)} cols)...")

    # Launch the load generator in a separate thread pool — monitor runs in the main thread
    # so we see the timeline in real time while load is being applied.
    pool = ThreadPoolExecutor(max_workers=args.n)
    futures = [pool.submit(_fire_one, endpoint, sample_df) for _ in range(args.n)]
    pool.shutdown(wait=False)

    try:
        monitor_timeline(args.endpoint_name, endpoint.boto3_session, args.minutes)
    except KeyboardInterrupt:
        print("\nInterrupted — cancelling pending futures...")
    finally:
        done = sum(1 for f in futures if f.done())
        print(f"\nInference requests done: {done}/{args.n}")


if __name__ == "__main__":
    main()
