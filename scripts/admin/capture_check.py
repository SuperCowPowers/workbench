"""Check that captured predictions match live endpoint inference.

This is a sanity check — when models/endpoints are recreated all captures are deleted,
so for a given endpoint + capture the stored predictions should exactly match what the
endpoint produces when re-run on the same compounds.

Usage:
    python scripts/admin/capture_check.py
"""

from workbench.api import Endpoint, Model, FeatureSet

# ---- Configuration (hardcoded) ----
ENDPOINT_NAME = "ppb-human-free-reg-chemprop-1-ts"
CAPTURE_NAME = "ts_20200211"

# Columns to compare
COMPARE_COLS = ["prediction", "prediction_std", "confidence"]
ID_COL = "udm_mol_bat_id"


def main():
    print(f"Endpoint: {ENDPOINT_NAME}")
    print(f"Capture:  {CAPTURE_NAME}")
    print("=" * 60)

    # Step 1: Get captured predictions
    end = Endpoint(ENDPOINT_NAME)
    model_name = end.get_input()
    _model = Model(model_name)
    captured_df = _model.get_inference_predictions(CAPTURE_NAME)
    print(f"\nCaptured predictions: {len(captured_df)} rows")

    # Step 2: Run live inference on the same compounds
    eval_ids = captured_df[ID_COL].tolist()
    fs = FeatureSet(_model.get_input())
    full_df = fs.pull_dataframe()
    eval_df = full_df[full_df[ID_COL].isin(eval_ids)]
    print(f"Eval compounds found in FeatureSet: {len(eval_df)}")

    print(f"\nRunning live inference on {ENDPOINT_NAME}...")
    live_df = end.inference(eval_df, capture_name=CAPTURE_NAME)
    print(f"Live predictions: {len(live_df)} rows")

    # Step 3: Merge and compare
    captured = captured_df[[ID_COL] + COMPARE_COLS].copy()
    live = live_df[[ID_COL] + COMPARE_COLS].copy()
    merged = captured.merge(live, on=ID_COL, suffixes=("_captured", "_live"))
    print(f"Matched rows: {len(merged)}")
    print("=" * 60)

    # Step 4: Check each column
    all_match = True
    for col in COMPARE_COLS:
        cap_col = f"{col}_captured"
        live_col = f"{col}_live"
        diff = (merged[cap_col] - merged[live_col]).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        exact_matches = (diff < 1e-10).sum()
        close_matches = (diff < 1e-6).sum()

        print(f"\n--- {col} ---")
        print(f"  Max |diff|:      {max_diff:.10f}")
        print(f"  Mean |diff|:     {mean_diff:.10f}")
        print(f"  Exact matches:   {exact_matches}/{len(merged)}")
        print(f"  Close (<1e-6):   {close_matches}/{len(merged)}")

        if max_diff > 1e-6:
            all_match = False
            # Show worst mismatches
            worst = merged.nlargest(5, diff.name if hasattr(diff, "name") else 0)
            # Re-compute for display
            merged["_diff"] = diff
            worst = merged.nlargest(5, "_diff")
            print("  Top mismatches:")
            for _, row in worst.iterrows():
                print(
                    f"    {row[ID_COL]:>15s}  captured={row[cap_col]:.6f}  "
                    f"live={row[live_col]:.6f}  diff={row['_diff']:.6f}"
                )
            merged.drop(columns=["_diff"], inplace=True)

    # Summary
    print("\n" + "=" * 60)
    if all_match:
        print("PASS: Captured predictions match live inference.")
    else:
        print("FAIL: Captured predictions differ from live inference!")
        print("      This may indicate the endpoint was updated without clearing captures.")


if __name__ == "__main__":
    main()
