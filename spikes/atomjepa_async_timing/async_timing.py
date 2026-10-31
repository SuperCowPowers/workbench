"""Which async knob moves GPU throughput: invocation batch size or the client poll schedule?

Times one Atom-JEPA GPU async endpoint over a fixed compound set, setting the batch size through
workbench_meta["inference_batch_size"] and the poll schedule through the async_inference module
constants. Each condition repeats in shuffled order, so drift over the sweep lands on all of them.

deploy copies the source model and deploys the copy. A model package pins the image digest it was
registered with, and a copy resolves the current :latest, so this reaches a rebuilt image without
retraining and leaves the source endpoint and its captures alone.

    python spikes/atomjepa_async_timing/async_timing.py deploy
    python spikes/atomjepa_async_timing/async_timing.py sweep --version 0.8.514
    python spikes/atomjepa_async_timing/async_timing.py teardown
"""

import argparse
import importlib
import logging
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from workbench.api import Endpoint, FeatureSet, Model
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.core.cloud_platform.aws.aws_meta import variant_instance

# importlib because the package also has a function named async_inference
async_module = importlib.import_module("workbench.endpoints.async_inference")

SOURCE_MODEL = "cyp-reg-atomjepa-3a4-readout"
NAME = "atomjepa-async-timing"
ROWS = 256
WARM_ROWS = 32
OUTPUT = Path(__file__).parent / "timing_results.csv"

POLL_SCHEDULES = {
    "default": (async_module._POLL_INITIAL_S, async_module._POLL_MAX_S),
    "fast": (0.5, 0.5),
}
CONDITIONS = [(16, "default"), (64, "default"), (128, "default"), (16, "fast"), (64, "fast")]


class BatchSizeLog(logging.Handler):
    """Records the batch size async_inference reports, which proves the meta knob took effect."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.batch_size = None

    def emit(self, record: logging.LogRecord):
        match = re.search(r"async_inference: .*batch_size=(\d+)", record.getMessage())
        if match:
            self.batch_size = int(match.group(1))


def serving_instance(sm) -> str:
    """The instance pool the endpoint is serving from."""
    endpoint = sm.describe_endpoint(EndpointName=NAME)
    config = sm.describe_endpoint_config(EndpointConfigName=endpoint["EndpointConfigName"])
    return variant_instance(config["ProductionVariants"][0], endpoint["ProductionVariants"][0])


def log_events(logs, pattern: str, start_ms: int = 0) -> list[dict]:
    """The endpoint's container log events matching a pattern, oldest first."""
    log_group = f"/aws/sagemaker/Endpoints/{NAME}"
    pages = logs.get_paginator("filter_log_events").paginate(
        logGroupName=log_group, filterPattern=pattern, startTime=start_ms
    )
    return sorted((e for page in pages for e in page["events"]), key=lambda e: e["timestamp"])


def require_version(logs, version: str):
    """Stop unless the newest instance logged the expected Workbench version at startup."""
    for _ in range(12):
        events = log_events(logs, '"Workbench Version"')
        if events:
            running = re.search(r"Workbench Version: ([\w.+]+)", events[-1]["message"]).group(1)
            if running != version:
                raise SystemExit(f"Endpoint runs Workbench {running}, not {version}: timing would mean nothing")
            print(f"Workbench version in the startup log: {running}")
            return
        time.sleep(10)
    raise SystemExit("No Workbench Version line in the endpoint log")


def set_batch_size(end: Endpoint, batch_size: int):
    """Set the meta knob and wait until a read returns it."""
    end.upsert_workbench_meta({"inference_batch_size": batch_size})
    for _ in range(30):
        if int((end.workbench_meta() or {}).get("inference_batch_size", 0)) == batch_size:
            return
        time.sleep(1)
    raise SystemExit(f"inference_batch_size never read back as {batch_size}")


def phases(logs, start_ms: int) -> pd.DataFrame:
    """Per invocation, from the container's own log lines during the sweep.

    predict_fn prints one line when conformers are built and one when scoring ends, and the
    endpoint runs one invocation at a time, so a pair brackets the fold forwards plus UQ, and the
    gap to the next pair is conformer generation plus any time the instance sat without work.
    """
    built = log_events(logs, '"Conformers for"', start_ms)
    complete = log_events(logs, '"Inference complete"', start_ms)
    rows = []
    for i, (b, c) in enumerate(zip(built, complete)):
        molecules = int(re.search(r"of (\d+) molecules", b["message"]).group(1))
        gap = (built[i + 1]["timestamp"] - c["timestamp"]) / 1000 if i + 1 < len(built) else np.nan
        rows.append({"rows": molecules, "forward_uq_s": (c["timestamp"] - b["timestamp"]) / 1000, "gap_to_next_s": gap})
    return pd.DataFrame(rows)


def report(results: pd.DataFrame, invocations: pd.DataFrame):
    """Seconds per row by condition, the per-invocation overhead fit, and the phase split."""
    results = results.assign(s_per_row=results["seconds"] / results["rows"])
    print("\nSeconds per row")
    print(results.groupby(["batch_size", "poll"])["s_per_row"].agg(["mean", "min", "max", "count"]).round(3))

    # seconds = overhead * chunks + the rest; the batch only changes the chunk count
    default = results[results["poll"] == "default"]
    if default["batch_size"].nunique() > 1:
        chunks = np.ceil(default["rows"] / default["batch_size"])
        overhead, rest = np.polyfit(chunks, default["seconds"], 1)
        print(f"\nPer-invocation overhead: {overhead:.2f} s; everything else: {rest / ROWS:.3f} s/row")
        for batch_size in sorted(default["batch_size"].unique()):
            share = (overhead / batch_size) / (overhead / batch_size + rest / ROWS)
            print(f"  batch {batch_size}: overhead is {share:.1%} of a row's cost")

    for batch_size in sorted(results.loc[results["poll"] == "fast", "batch_size"].unique()):
        means = results[results["batch_size"] == batch_size].groupby("poll")["seconds"].mean()
        print(f"Poll effect at batch {batch_size}: {means['default'] - means['fast']:+.1f} s per {ROWS}-row run")

    if not invocations.empty:
        print("\nContainer phases by invocation size, median seconds (1-row invocations are warmers)")
        print(invocations.groupby("rows").agg(["median", "count"]).round(2))


def sweep(version: str, reps: int):
    """Run every condition reps times in shuffled order, then report."""
    session = AWSAccountClamp().boto3_session
    sm, logs = session.client("sagemaker"), session.client("logs")
    end = Endpoint(NAME)
    model = Model(SOURCE_MODEL)

    df = FeatureSet(model.get_input()).pull_dataframe()[model.features()]
    df = df.sample(ROWS, random_state=0).reset_index(drop=True)

    batch_log = BatchSizeLog()
    logging.getLogger("workbench").addHandler(batch_log)

    # The throwaway run covers scale-from-zero and first-use model compilation
    print(f"Warming with {WARM_ROWS} rows...")
    set_batch_size(end, WARM_ROWS)
    end.fast_inference(df.head(WARM_ROWS))
    require_version(logs, version)

    schedule = [condition for _ in range(reps) for condition in random.sample(CONDITIONS, len(CONDITIONS))]
    start_ms = int(time.time() * 1000)
    results, reference = [], None
    for run, (batch_size, poll) in enumerate(schedule, 1):
        set_batch_size(end, batch_size)
        async_module._POLL_INITIAL_S, async_module._POLL_MAX_S = POLL_SCHEDULES[poll]
        batch_log.batch_size = None

        start = time.time()
        output = end.fast_inference(df)
        seconds = time.time() - start

        if batch_log.batch_size != batch_size:
            raise SystemExit(f"Asked for batch {batch_size}, async_inference ran {batch_log.batch_size}")
        predictions = output["prediction"].to_numpy()
        reference = predictions if reference is None else reference
        results.append(
            {
                "run": run,
                "batch_size": batch_size,
                "poll": poll,
                "rows": len(df),
                "seconds": round(seconds, 1),
                "scored": int(np.isfinite(predictions).sum()),
                "max_prediction_shift": float(np.nanmax(np.abs(predictions - reference))),
                "instance": serving_instance(sm),
            }
        )
        print(results[-1])
        pd.DataFrame(results).to_csv(OUTPUT, index=False)

    results = pd.DataFrame(results)
    if results["instance"].nunique() > 1:
        print(f"WARNING: the sweep crossed instance pools: {sorted(results['instance'].unique())}")
    if (results["scored"] < ROWS).any():
        print("WARNING: some runs lost rows; those timings exclude failed chunks")
    # CloudWatch ingestion trails the container
    time.sleep(30)
    report(results, phases(logs, start_ms))
    print(f"\nResults: {OUTPUT}")


def deploy():
    model = Model(SOURCE_MODEL).copy(NAME)
    model.to_endpoint(name=NAME, tags=[NAME], async_endpoint=True)


def teardown():
    Endpoint(NAME).delete()
    Model(NAME).delete()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("deploy")
    commands.add_parser("teardown")
    sweep_parser = commands.add_parser("sweep")
    sweep_parser.add_argument("--version", required=True, help="Workbench version the endpoint must be running")
    sweep_parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args()

    if args.command == "deploy":
        deploy()
    elif args.command == "teardown":
        teardown()
    else:
        sweep(args.version, args.reps)


if __name__ == "__main__":
    main()
