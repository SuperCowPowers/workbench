"""End-to-end AWS smoke: ChemProp with a designated validation set.

Trains a real model with ``validation_ids`` and confirms the held-out rows are
(1) retained + marked in the training view and (2) scored, never trained. The
ChemProp held-out path runs a fresh ensemble-inference pass over the held-out
SMILES. Uses a molecular FeatureSet (aqsol) since ChemProp requires SMILES.

Note: ChemProp drops invalid SMILES, so the scored held-out count may be < the
number of validation_ids (all designated rows are still retained in the view).
"""

import pytest

# Workbench Imports
from workbench.api import FeatureSet, Model, ModelFramework
from workbench.core.artifacts.model_core import ModelType


def _assert_holdout(model: Model, id_column: str, validation_ids: list):
    """Shared assertions: view marks + retains the held-out rows, and they get scored."""
    # 1) The training view retains the validation rows and marks them
    view_df = model.training_view().pull_dataframe()
    assert {"validation", "exclude"}.issubset(view_df.columns), "view missing role columns"
    view_ids = set(view_df[id_column].tolist())
    assert set(validation_ids).issubset(view_ids), "validation rows were dropped from the view"
    val_flag = view_df.set_index(id_column)["validation"].astype(bool)
    assert val_flag.loc[validation_ids].all(), "validation rows not marked validation=True"
    non_val = [i for i in view_ids if i not in set(validation_ids)]
    assert not val_flag.loc[non_val].any(), "non-validation rows incorrectly marked"

    # 2) The held-out rows were scored (present + non-null) in validation_predictions.csv
    preds = model.get_inference_predictions("model_training")
    assert preds is not None and "validation" in preds.columns, "validation_predictions missing validation column"
    held = preds[preds["validation"].astype(bool)]
    assert len(held) > 0, "no held-out rows in validation_predictions"
    assert held["prediction"].notna().all(), "held-out rows were not scored"


@pytest.mark.long
def test_chemprop_validation_set():
    fs = FeatureSet("aqsol_features")
    id_column = fs.id_column
    validation_ids = fs.pull_dataframe()[id_column].tolist()[:50]

    name = "aqsol-chemprop-validation-test"
    model = fs.to_model(
        name=name,
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="solubility",
        feature_list=["smiles"],
        tags=["test", "aqsol", "validation-test"],
        description="AqSol ChemProp (validation-set smoke)",
        validation_ids=validation_ids,
    )
    try:
        _assert_holdout(model, id_column, validation_ids)
    finally:
        model.delete()


if __name__ == "__main__":
    test_chemprop_validation_set()
