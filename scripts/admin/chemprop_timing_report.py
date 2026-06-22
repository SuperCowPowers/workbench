"""Pull and tabulate training-timing for Chemprop SageMaker training jobs.

Reads the ``[timing]`` / ``[cache]`` markers emitted by the chemprop training
template. For each model/job name given, finds the latest SageMaker training job,
extracts those markers from its CloudWatch log, pulls GPU/CPU/memory utilization,
and prints a side-by-side comparison.

Args are model names or training-job name prefixes (one or more).

Usage:
    python scripts/admin/chemprop_timing_report.py <name> [<name> ...]
    AWS_PROFILE=<profile> python scripts/admin/chemprop_timing_report.py my-model-1 my-model-2
"""

import re
import sys
from datetime import datetime, timezone

import boto3

LOG_GROUP = "/aws/sagemaker/TrainingJobs"
METRIC_NS = "/aws/sagemaker/TrainingJobs"


def latest_job(sm, prefix: str):
    """Most recent training job whose name contains `prefix` (None if none)."""
    resp = sm.list_training_jobs(NameContains=prefix, SortBy="CreationTime", SortOrder="Descending", MaxResults=5)
    summaries = resp.get("TrainingJobSummaries", [])
    return summaries[0] if summaries else None


def timing_lines(logs, job_name: str):
    """All [timing] / [cache] log lines for the job, in order."""
    streams = logs.describe_log_streams(logGroupName=LOG_GROUP, logStreamNamePrefix=job_name).get("logStreams", [])
    if not streams:
        return []
    stream = streams[0]["logStreamName"]
    out, token = [], None
    while True:
        kw = dict(
            logGroupName=LOG_GROUP,
            logStreamNames=[stream],
            filterPattern='?"[timing]" ?"[cache]" ?"template revision"',
            limit=1000,
        )
        if token:
            kw["nextToken"] = token
        resp = logs.filter_log_events(**kw)
        out += [e["message"].strip() for e in resp.get("events", [])]
        token = resp.get("nextToken")
        if not token:
            break
    return out


def metric(cw, job_name: str, name: str, start, end):
    """(peak, mean) for a SageMaker training metric over the job window, or (None, None)."""
    resp = cw.get_metric_statistics(
        Namespace=METRIC_NS,
        MetricName=name,
        Dimensions=[{"Name": "Host", "Value": f"{job_name}/algo-1"}],
        StartTime=start,
        EndTime=end,
        Period=60,
        Statistics=["Maximum", "Average"],
    )
    pts = resp.get("Datapoints", [])
    if not pts:
        return None, None
    peak = max(p["Maximum"] for p in pts)
    mean = sum(p["Average"] for p in pts) / len(pts)
    return peak, mean


def parse_metrics_from_timing(lines):
    """Pull a few headline numbers out of the [timing]/[cache] lines for the summary table."""
    out = {"cache_s": None, "total_train_s": None, "fold_fits": []}
    for ln in lines:
        m = re.search(r"\[cache\].*precomputed in ([\d.]+)s", ln)
        if m:
            out["cache_s"] = float(m.group(1))
        m = re.search(r"total training \(\d+ folds\): ([\d.]+)s", ln)
        if m:
            out["total_train_s"] = float(m.group(1))
        m = re.search(r"fold \d+ fit: ([\d.]+)s", ln)
        if m:
            out["fold_fits"].append(float(m.group(1)))
    return out


def main(model_names):
    sm = boto3.client("sagemaker")
    logs = boto3.client("logs")
    cw = boto3.client("cloudwatch")

    summary = []
    for name in model_names:
        print("\n" + "=" * 70)
        print(f"MODEL: {name}")
        print("=" * 70)

        job = latest_job(sm, name)
        if job is None:
            print("  no training job found")
            continue
        job_name = job["TrainingJobName"]
        desc = sm.describe_training_job(TrainingJobName=job_name)
        start = desc.get("TrainingStartTime")
        # InProgress jobs have no end time yet — use "now" so the metric window is valid.
        end = desc.get("TrainingEndTime") or datetime.now(timezone.utc)
        secs = desc.get("TrainingTimeInSeconds")
        inst = desc.get("ResourceConfig", {}).get("InstanceType")
        print(f"  job: {job_name}")
        print(
            f"  status: {job['TrainingJobStatus']}  instance: {inst}  "
            f"train_time: {secs}s" + (f" ({secs // 60}m)" if secs else "")
        )

        lines = timing_lines(logs, job_name)
        if lines:
            print("  --- timing log ---")
            for ln in lines:
                # strip CloudWatch's leading container-id/ansi noise if present
                print("   ", re.sub(r"^.*?(\[timing\]|\[cache\]|.*template revision)", r"\1", ln))
        else:
            print("  (no [timing]/[cache] lines — template may predate the instrumentation)")

        gpu_pk, gpu_mn = metric(cw, job_name, "GPUUtilization", start, end)
        gmem_pk, _ = metric(cw, job_name, "GPUMemoryUtilization", start, end)
        cpu_pk, cpu_mn = metric(cw, job_name, "CPUUtilization", start, end)
        mem_pk, _ = metric(cw, job_name, "MemoryUtilization", start, end)
        print("  --- utilization ---")

        def fmt(v):
            return f"{v:.0f}%" if v is not None else "n/a"

        print(f"    GPU util: peak {fmt(gpu_pk)} / mean {fmt(gpu_mn)}   GPU mem: peak {fmt(gmem_pk)}")
        print(f"    CPU util: peak {fmt(cpu_pk)} / mean {fmt(cpu_mn)}   RAM: peak {fmt(mem_pk)}")

        t = parse_metrics_from_timing(lines)
        mean_fit = sum(t["fold_fits"]) / len(t["fold_fits"]) if t["fold_fits"] else None
        summary.append(
            {
                "model": name,
                "cache_s": t["cache_s"],
                "mean_fit_s": mean_fit,
                "total_train_s": t["total_train_s"],
                "gpu_mean": gpu_mn,
                "gpu_peak": gpu_pk,
            }
        )

    # Summary table
    if summary:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        hdr = (
            f"{'model':<32} {'cache(s)':>9} {'mean fold fit(s)':>17} "
            f"{'total train(s)':>15} {'GPU mean':>9} {'GPU peak':>9}"
        )
        print(hdr)
        print("-" * len(hdr))
        for s in summary:

            def f(v, suf=""):
                return f"{v:.1f}{suf}" if isinstance(v, (int, float)) else "n/a"

            print(
                f"{s['model'][:32]:<32} {f(s['cache_s']):>9} {f(s['mean_fit_s']):>17} "
                f"{f(s['total_train_s']):>15} {f(s['gpu_mean'], '%'):>9} {f(s['gpu_peak'], '%'):>9}"
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
