"""Typical-usage async endpoint test.

Emulates what a real user does with an async endpoint — grab a dataframe, call
``inference()`` on the whole thing in one shot, get results back. No threading,
no chunking, no tuning. Everything comes from framework defaults.

A background thread prints the autoscaling timeline so you can watch 0→N→0
while the main call is blocking — but that's observation only, the user code
itself (the ``endpoint.inference(df)`` line) is what matters.

Usage:
    python scripts/admin/async_typical_usage.py <endpoint_name> [--rows 500]
"""

import argparse
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

from workbench.api import AsyncEndpoint, PublicData


def _get_metric(cw, endpoint_name: str, metric: str, stat: str, start, end):
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


def _monitor(endpoint_name: str, boto3_session, stop_event: threading.Event) -> None:
    """Background scaling monitor — prints a timeline every 30s until stopped."""
    cw = boto3_session.client("cloudwatch")
    sm = boto3_session.client("sagemaker")

    print(f"\n{'t':>6s}  {'instances':>9s}  {'has_backlog':>11s}  {'backlog_per_instance':>21s}")
    print("-" * 56)

    t_start = time.time()
    while not stop_event.is_set():
        t_rel = int(time.time() - t_start)
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=3)

        try:
            desc = sm.describe_endpoint(EndpointName=endpoint_name)
            instances = desc["ProductionVariants"][0].get("DesiredInstanceCount", 0)
        except Exception:
            instances = -1

        has_backlog = _get_metric(cw, endpoint_name, "HasBacklogWithoutCapacity", "Maximum", start, end)
        per_instance = _get_metric(cw, endpoint_name, "ApproximateBacklogSizePerInstance", "Average", start, end)

        hb = "--" if has_backlog is None else f"{has_backlog:.0f}"
        bp = "--" if per_instance is None else f"{per_instance:.1f}"
        print(f"{t_rel:>5d}s  {instances:>9d}  {hb:>11s}  {bp:>21s}")

        # stop_event.wait lets us exit faster than a plain sleep
        if stop_event.wait(timeout=30):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("endpoint_name", help="Name of the async endpoint")
    parser.add_argument("--rows", type=int, default=500, help="Rows from aqsol to run (default 500)")
    args = parser.parse_args()

    # -------- The "typical user" part --------------------------------------
    endpoint = AsyncEndpoint(args.endpoint_name)
    if not endpoint.exists():
        print(f"Endpoint '{args.endpoint_name}' not found.", file=sys.stderr)
        sys.exit(1)

    aqsol = PublicData().get("comp_chem/aqsol/aqsol_public_data")
    aqsol.columns = aqsol.columns.str.lower()
    df = aqsol[["smiles"]].head(args.rows).reset_index(drop=True)

    print(f"Running inference on {len(df)} SMILES via '{args.endpoint_name}'...")
    print("(Framework handles batching, parallel submission, and polling internally.)")

    # -------- Observational sidecar ----------------------------------------
    # The monitor thread is NOT part of typical user code — it's just here to
    # show the autoscaling behavior while the main inference() call blocks.
    stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor,
        args=(args.endpoint_name, endpoint.boto3_session, stop),
        daemon=True,
        name="scaling-monitor",
    )
    monitor.start()

    # -------- The one line that matters ------------------------------------
    t0 = time.time()
    try:
        results = endpoint.inference(df)
    except KeyboardInterrupt:
        stop.set()
        print("\nInterrupted.")
        raise

    elapsed = int(time.time() - t0)
    # One more monitor tick so the final "instances=0" (or close to it) shows up
    time.sleep(30)
    stop.set()
    monitor.join(timeout=5)

    print(f"\nDone. {len(results)} rows returned in {elapsed}s ({elapsed / 60:.1f} min).")
    print(f"Columns returned: {list(results.columns)[:8]}{'...' if len(results.columns) > 8 else ''}")


if __name__ == "__main__":
    main()
