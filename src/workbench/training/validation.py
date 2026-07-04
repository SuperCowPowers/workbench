"""Validation-set helpers for model training scripts.

Training-only per the :mod:`workbench.training` contract — templates import this
**only inside their ``__main__``** (deferred), never at top level.

The per-model training view carries a boolean ``validation`` column marking
held-out rows (see ``view_utils.create_model_training_view``). These rows are
kept in the training CSV but must not train; the model script routes them out of
the train/CV path and scores them as an honest held-out set.
"""

import pandas as pd


def split_validation_set(df: pd.DataFrame, marker: str = "validation") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training DataFrame into (train, held-out validation) frames.

    The validation frame is the rows where ``marker`` is truthy. When the column
    is absent or nothing is marked, the validation frame is empty and the train
    frame is the input unchanged — so callers can adopt this unconditionally and
    it's a no-op for models without a designated validation set.

    Both returned frames have a reset index.

    Args:
        df (pd.DataFrame): The loaded training data (from the model training view).
        marker (str): Boolean column marking held-out rows (default "validation").

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df).
    """
    if marker not in df.columns:
        return df.reset_index(drop=True), df.iloc[0:0].reset_index(drop=True)

    is_val = df[marker].fillna(False).astype(bool)
    train_df = df[~is_val].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)
    return train_df, val_df
