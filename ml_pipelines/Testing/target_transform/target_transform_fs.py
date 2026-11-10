"""Build the as-measured MPPB FeatureSet used by the target-transform tests.

The OpenADMET `open_admet_mppb` FeatureSet carries a log-transformed target, which cannot
exercise a model-side `target_transform`. This one keeps MPPB as the assay reports it:
percent unbound, bounded [0, 100], with 115 of 1302 rows under 1% and 10 exact zeros.

`mppb_floored` carries the same target with those zeros clamped to the assay's smallest
reading. A `logit` needs 0 < y < ceiling and clamps nothing itself, so the clamp belongs
here, in the data, where it stays visible. `log10` and `arcsinh` train on `mppb` directly.

DataSources:
    - mppb_raw
FeatureSets:
    - mppb_raw
"""

from workbench.api import DataSource, Endpoint, FeatureSet, PublicData

PUBLIC_DATA = "comp_chem/openadmet/expansionrx/train/mppb"
NAME = "mppb_raw"
FEATURE_ENDPOINT = "smiles-to-2d-v1"
TARGET = "mppb"
FLOORED_TARGET = "mppb_floored"
LLOQ = 0.1  # the assay's smallest non-zero reading


def main():
    df = PublicData().get(PUBLIC_DATA).rename(columns={"id": "molecule_name"})
    df = df.dropna(subset=["mppb"])
    print(f"Loaded {len(df)} MPPB rows: {df['mppb'].min():g} to {df['mppb'].max():g} percent unbound")
    print(f"  under 1%: {(df['mppb'] < 1).sum()}, exact zeros: {(df['mppb'] == 0).sum()}")

    df[FLOORED_TARGET] = df[TARGET].clip(lower=LLOQ)
    print(f"  {FLOORED_TARGET}: min {df[FLOORED_TARGET].min():g} (for logit models)")

    # 2D descriptors for the tabular frameworks; chemprop and atomjepa read `smiles`
    df = Endpoint(FEATURE_ENDPOINT).inference(df)

    DataSource(df, name=NAME)
    DataSource(NAME).to_features(NAME, id_column="molecule_name", tags=["target_transform", "mppb"])
    print(f"FeatureSet {NAME}: {FeatureSet(NAME).num_rows()} rows")


if __name__ == "__main__":
    main()
