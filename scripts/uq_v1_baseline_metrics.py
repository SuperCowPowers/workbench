#!/usr/bin/env python3
"""Capture pre-conformal-change baseline metrics for V1 UQ models.

For each V1 model:
  - Pull cross-fold inference from its deployed endpoint (with quantiles).
  - Compute ``uq_metrics``.
  - Save both the metrics and the raw dataframe to a timestamped snapshot dir.

After the locally-adaptive-conformal change ships and the same model names are
retrained + redeployed, re-run this script. The two snapshot directories then
support a direct before/after diff (same model names, same metric definitions,
same data source).

Usage:
    python scripts/uq_v1_baseline_metrics.py [model_name ...] [--output PATH]

If no model names are given, falls back to the current V1 trio:
    logd-reg-chemprop-new-uq, aqsol-reg-pytorch-new-uq, aqsol-regression-new-uq

Output layout:
    scripts/snapshots/v1_baseline_<timestamp>/
      ├── metrics.json                # {model: {endpoint, target, n_rows, metrics}, ...}
      └── <model_name>.parquet        # raw cross-fold inference df, one per model
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from workbench.api import Endpoint, Model
from workbench.utils.model_utils import uq_metrics

log = logging.getLogger("workbench")


DEFAULT_MODELS = [
    "logd-reg-chemprop-new-uq",
    "aqsol-reg-pytorch-new-uq",
    "aqsol-regression-new-uq",
]


def snapshot_model(model_name: str, out_dir: Path) -> dict:
    """Capture metrics + raw dataframe for one V1 model."""
    log.info(f"Snapshotting {model_name}...")
    model = Model(model_name)

    target = model.target()
    if isinstance(target, list):
        if len(target) != 1:
            raise RuntimeError(
                f"Model {model_name} has multi-target ({target}); uq_metrics expects a single target column"
            )
        target = target[0]
    if target is None:
        raise RuntimeError(f"Model {model_name} has no target column registered")

    endpoint_names = model.endpoints()
    if not endpoint_names:
        raise RuntimeError(f"Model {model_name} has no deployed endpoints")
    endpoint_name = endpoint_names[0]
    endpoint = Endpoint(endpoint_name)

    # include_quantiles=True is critical — uq_metrics needs q_025/q_975/etc.
    # to compute coverage and width metrics. Defaults to False on the API.
    df = endpoint.cross_fold_inference(include_quantiles=True)

    # Save raw df for posterity (cheap, lets us recompute metrics later
    # without re-hitting the endpoint).
    df_path = out_dir / f"{model_name}.parquet"
    df.to_parquet(df_path, index=False)
    log.info(f"  Saved raw df ({len(df)} rows, {len(df.columns)} cols) to {df_path}")

    metrics = uq_metrics(df, target_col=target)
    log.info(f"  Computed {len(metrics)} metrics")

    return {
        "model_name": model_name,
        "endpoint_name": endpoint_name,
        "target": target,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "models",
        nargs="*",
        default=DEFAULT_MODELS,
        help=f"V1 model names to snapshot (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory (default: scripts/snapshots/v1_baseline_<timestamp>)",
    )
    args = parser.parse_args()

    if args.output:
        out_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = Path(__file__).resolve().parent / "snapshots" / f"v1_baseline_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== V1 baseline metrics snapshot ===")
    print(f"  Output: {out_dir}")
    print(f"  Models: {args.models}")
    print()

    results: dict[str, dict] = {}
    failures: list[str] = []
    for name in args.models:
        try:
            results[name] = snapshot_model(name, out_dir)
            print(f"  OK   {name}")
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
            failures.append(name)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print()
    print(f"Wrote metrics: {metrics_path}")

    if failures:
        print(f"Failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
