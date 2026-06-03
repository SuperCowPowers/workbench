"""Manual integration check: NaT in a (non-event-time) date column survives FeatureSet ingestion.

This is the one thing the offline unit tests can't cover — whether AWS Feature
Store / Athena accept a null in the `udm_asy_date` string column (date-less
public rows) and whether it reads back as NaT so a temporal split keeps it on the
training side.

Run with:
    WORKBENCH_CONFIG=/Users/briford/.workbench/scp_sandbox.json \
        python tests/specific/multi_task_nat_ingest_check.py

Creates and then deletes a throwaway FeatureSet 'test_mt_nat_ingest'.
"""

import pandas as pd

from workbench.api import FeatureSet
from workbench.core.transforms.pandas_transforms import PandasToFeatures

FS_NAME = "test_mt_nat_ingest"
CUTOFF = "2025-10-17"


def main():
    # 3 dated rows (one post-cutoff) + 2 date-less rows (NaT), mimicking the
    # logp_public smiles-only rows in the v2 multi-task merge.
    df = pd.DataFrame(
        {
            "udm_mol_bat_id": ["1", "2", "3", "pub_a", "pub_b"],
            "smiles": ["CCO", "CCC", "CCCC", "c1ccccc1", "CCN"],
            "ppb_human": [10.0, 20.0, 30.0, float("nan"), float("nan")],
            "logp": [1.0, float("nan"), float("nan"), 2.0, 3.0],
            # Plain YYYY-MM-DD strings with NaN for the date-less public rows —
            # this is exactly what pull_multi_task_data now emits.
            "udm_asy_date": ["2024-01-01", "2026-01-01", "2024-06-01", None, None],
        }
    )
    print("INPUT date nulls:", df["udm_asy_date"].isna().tolist())

    to_features = PandasToFeatures(FS_NAME)
    to_features.set_input(df, id_column="udm_mol_bat_id")
    to_features.set_output_tags(["test", "multi-task", "nat-check"])
    to_features.transform()

    fs = FeatureSet(FS_NAME)
    back = fs.pull_dataframe()
    print(f"\nRead back {len(back)} rows; columns: {sorted(back.columns)}")

    # 1. Date nulls round-trip: the two public rows must come back null.
    nulls = back.set_index("udm_mol_bat_id")["udm_asy_date"].isna().to_dict()
    print("READBACK date null by id:", nulls)
    assert nulls.get("pub_a") and nulls.get("pub_b"), "date-less rows did not round-trip as null!"
    assert not nulls.get("1") and not nulls.get("2"), "dated rows came back null!"

    # 2. Temporal split: only the post-cutoff dated row is held out; NaT rows train.
    holdout = fs.temporal_split("udm_asy_date", end_date=CUTOFF)
    holdout_ids = set(holdout)
    print("HOLDOUT ids:", holdout_ids)
    assert holdout_ids == {"2"}, f"expected only id '2' in holdout, got {holdout_ids}"

    print("\nPASS: NaT dates ingest, round-trip as null, and fall on the training side.")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Always clean up the throwaway FeatureSet.
        try:
            fs = FeatureSet(FS_NAME)
            if fs.exists():
                print(f"\nDeleting throwaway FeatureSet '{FS_NAME}'...")
                fs.delete()
        except Exception as e:  # noqa: BLE001
            print(f"Cleanup warning: {e}")
