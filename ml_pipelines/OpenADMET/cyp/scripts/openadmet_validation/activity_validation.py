from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pathlib import Path

ACTIVITY_DATASET_SIZE = 750
CYP_ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
ACTIVITY_VALUE_COLUMNS = tuple(f"{cyp}_pIC50_direct_inhibition" for cyp in CYP_ISOFORMS)


def _as_set(values: Iterable[str]) -> set[str]:
    return {str(v) for v in values}


def validate_activity_submission(
    activity_predictions_file: Path,
    expected_ids: set[str] | None = None,
    required_id_columns: tuple[str, ...] = ("SMILES", "Molecule_Name"),
    required_value_columns: tuple[str, ...] = ACTIVITY_VALUE_COLUMNS,
) -> tuple[bool, list[str]]:
    errors: list[str] = []

    path = Path(activity_predictions_file)
    if not path.exists():
        return False, [f"File does not exist: {path}"]

    try:
        activity_predictions = pd.read_csv(path)
    except Exception as exc:
        return False, [f"Error reading CSV file: {exc}"]

    required_columns = (*required_id_columns, *required_value_columns)
    missing_columns = [col for col in required_columns if col not in activity_predictions.columns]
    if missing_columns:
        errors.append(f"Missing required column(s): {missing_columns}")
        return False, errors

    if activity_predictions.empty:
        errors.append("Submission is empty.")
        return False, errors

    null_id_rows = activity_predictions[list(required_id_columns)].isna().any(axis=1).sum()
    if null_id_rows:
        errors.append(f"Found {null_id_rows} row(s) with missing identifier values.")

    if "Molecule_Name" in activity_predictions.columns:
        duplicate_ids = activity_predictions["Molecule_Name"].duplicated().sum()
        if duplicate_ids:
            errors.append(f"Found {duplicate_ids} duplicated 'Molecule_Name' value(s).")

    for col in required_value_columns:
        numeric_col = pd.to_numeric(activity_predictions[col], errors="coerce")
        invalid_numeric = numeric_col.isna().sum()
        if invalid_numeric:
            errors.append(f"Column '{col}' contains {invalid_numeric} non-numeric or missing value(s).")
            continue

        non_finite = (~np.isfinite(numeric_col.to_numpy())).sum()
        if non_finite:
            errors.append(f"Column '{col}' contains {non_finite} non-finite value(s) (inf or -inf).")

    submitted_ids = _as_set(activity_predictions["Molecule_Name"])
    if expected_ids is not None:
        expected_ids = _as_set(expected_ids)
        missing = sorted(expected_ids - submitted_ids)
        extra = sorted(submitted_ids - expected_ids)
        if missing:
            errors.append(f"Missing {len(missing)} expected molecule(s): {missing[:20]}")
        if extra:
            errors.append(f"Found {len(extra)} unexpected molecule(s): {extra[:20]}")
    elif len(activity_predictions) != ACTIVITY_DATASET_SIZE:
        errors.append(
            f"Submission contains {len(activity_predictions)} rows, expected {ACTIVITY_DATASET_SIZE}."
        )

    return len(errors) == 0, errors
