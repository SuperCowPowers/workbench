"""Create the 'logp_public' FeatureSet from the merged public LogP dataset.

Pulls comp_chem/logp/logp_all (already published to the workbench-public-data
bucket) and rolls it straight into a FeatureSet via PandasToFeatures.

Run:
    python build_logp_public.py
"""

import logging

from workbench.api import PublicData
from workbench.core.transforms.pandas_transforms import PandasToFeatures

log = logging.getLogger("workbench")

FEATURE_SET_NAME = "logp_public"
ID_COL = "id"
TAGS = ["logp", "public"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    df = PublicData().get("comp_chem/logp/logp_all")
    log.info(f"Pulled {len(df):,} rows, columns={list(df.columns)}")

    log.info(f"Creating FeatureSet '{FEATURE_SET_NAME}' …")
    to_features = PandasToFeatures(FEATURE_SET_NAME)
    to_features.set_input(df, id_column=ID_COL)
    to_features.set_output_tags(TAGS)
    to_features.transform()
    log.info(f"Done. FeatureSet '{FEATURE_SET_NAME}' is ready.")


if __name__ == "__main__":
    main()
