"""Pull an endpoint's SageMaker data capture and report on what came back.

Confirms the capture config is sound and the payloads parse cleanly. An endpoint whose
capture config omits the content type header stores everything base64-encoded, which
shows up here as skipped payloads and an empty frame.

Usage:
    python scripts/admin/data_capture_report.py <endpoint-name>
    python scripts/admin/data_capture_report.py <endpoint-name> --from-date 2026-05-01
"""

import argparse

import pandas as pd

from workbench.api import Endpoint


def report_frame(label: str, df: pd.DataFrame):
    """Print shape, block count, timestamp span, and null-heavy columns for a captured frame."""
    print(f"\n--- {label} ---")
    if df.empty:
        print("  no rows")
        return

    print(f"  rows/cols:     {df.shape[0]} x {df.shape[1]}")

    # A consolidated frame holds one block per dtype; more than that is fragmented
    # and slow for anything downstream
    blocks, dtypes = len(df._mgr.blocks), df.dtypes.nunique()
    verdict = "fragmented" if blocks > dtypes else "consolidated"
    print(f"  blocks:        {blocks} for {dtypes} dtypes ({verdict})")

    if "timestamp" in df:
        stamps = df["timestamp"]
        print(f"  timestamp:     {stamps.min()} -> {stamps.max()}  dtype={stamps.dtype}")
        if stamps.isna().any():
            print(f"  MISSING stamps: {stamps.isna().sum()}")
    else:
        print("  timestamp:     absent")

    # Columns that are entirely null usually mean schema drift across the capture window
    all_null = [c for c in df.columns if df[c].isna().all()]
    mostly_null = [c for c in df.columns if 0.5 < df[c].isna().mean() < 1.0]
    print(f"  all-null cols:    {len(all_null)}  {all_null[:6]}")
    print(f"  >50% null cols:   {len(mostly_null)}  {mostly_null[:6]}")
    counts = {name: int(n) for name, n in df.dtypes.astype(str).value_counts().items()}
    print(f"  dtypes:        {counts}")


def main(endpoint_name: str, from_date: str):
    end = Endpoint(endpoint_name)
    dc = end.data_capture()

    print(f"Endpoint: {endpoint_name}")
    print(f"Enabled:  {dc.is_enabled()}")
    if not dc.is_enabled():
        print("Data capture is not enabled on this endpoint.")
        return

    config = dc.get_config() or {}
    content_types = config.get("CaptureContentTypeHeader")
    print(f"Path:     {dc.data_capture_path}")
    print(f"Content types: {content_types or 'NOT SET -- payloads will be stored base64'}")

    input_df, output_df = dc.get_captured_data(from_date=from_date)
    report_frame("input", input_df)
    report_frame("output", output_df)

    if not input_df.empty and not output_df.empty:
        print(f"\ninput/output row counts {'match' if len(input_df) == len(output_df) else 'DIFFER'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("endpoint_name", help="name of the endpoint to pull captures from")
    parser.add_argument("--from-date", help="only process captures from this date onward (YYYY-MM-DD)")
    args = parser.parse_args()
    main(args.endpoint_name, args.from_date)
