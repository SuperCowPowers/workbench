"""This Script creates three Chemprop models that differ only in UQ version

Same FeatureSet, target, framework, and held-out validation rows — the only difference
is the ``uq_version`` hyperparameter, so the three endpoints are directly comparable on
their confidence and quantile columns. The validation set is a Bemis-Murcko scaffold
hold-out: those scaffolds never appear in training, so the rows are genuinely
out-of-distribution and the v0/v1/v2 differences actually separate. Each endpoint
then scores those same rows through the serving path, captured as
``scaffold_holdout``.

Requires the open_admet_logd FeatureSet (ml_pipelines/OpenADMET/load_data).

Models:
    - logd-chemprop-uq-v0
    - logd-chemprop-uq-v1
    - logd-chemprop-uq-v2

Endpoints:
    - logd-chemprop-uq-v0
    - logd-chemprop-uq-v1
    - logd-chemprop-uq-v2
"""

import logging

from sklearn.model_selection import GroupShuffleSplit

# Workbench Imports
from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType
from workbench.training.splits import get_scaffold_groups

log = logging.getLogger("workbench")

FS_NAME = "open_admet_logd"
TARGET = "logd"
ID_COLUMN = "molecule_name"

# v0: isotonic on (prediction, std). v1 (default): proximity-augmented RF error model.
# v2: pure applicability domain from fingerprint neighbors.
UQ_VERSIONS = ["v0", "v1", "v2"]

# Fraction of scaffold groups held out of training and scored as the validation set
VALIDATION_FRACTION = 0.2
SEED = 42


def scaffold_validation_ids(fs: FeatureSet) -> list:
    """Pick a scaffold-disjoint hold-out set — no training scaffold appears in it."""
    df = fs.pull_dataframe()[[ID_COLUMN, "smiles"]].sort_values(ID_COLUMN)
    groups = get_scaffold_groups(df["smiles"].tolist())
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=SEED)
    _, val_idx = next(splitter.split(df, groups=groups))
    return df.iloc[val_idx][ID_COLUMN].tolist()


if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # All three models share these ids, otherwise the UQ columns aren't comparable
    fs = FeatureSet(FS_NAME)
    validation_ids = scaffold_validation_ids(fs)
    log.important(f"Scaffold hold-out: {len(validation_ids)} validation rows")

    # The held-out rows themselves, scored through each endpoint below
    val_df = fs.pull_dataframe()
    val_df = val_df[val_df[ID_COLUMN].isin(validation_ids)]

    for version in UQ_VERSIONS:
        name = f"logd-chemprop-uq-{version}"
        tags = ["open_admet", "chemprop", "uq_compare", version]

        if recreate or not Model(name).exists():
            m = fs.to_model(
                name=name,
                model_type=ModelType.UQ_REGRESSOR,
                model_framework=ModelFramework.CHEMPROP,
                target_column=TARGET,
                feature_list=["smiles"],
                description=f"LogD Chemprop Regression Model ({version.upper()} UQ)",
                tags=tags,
                hyperparameters={"uq_version": version},
                validation_ids=validation_ids,
            )
            m.set_owner("test")

        if recreate or not Endpoint(name).exists():
            end = Model(name).to_endpoint(tags=tags)
            end.set_owner("test")
            end.test_inference()
            end.cross_fold_inference()

            # Score the scaffold hold-out through the real serving path. Same rows,
            # same capture name across all three, so the confidence and q_* columns
            # line up row-for-row when comparing versions.
            end.inference(val_df, capture_name="scaffold_holdout", include_quantiles=True)

    log.important("UQ compare model creation complete.")
