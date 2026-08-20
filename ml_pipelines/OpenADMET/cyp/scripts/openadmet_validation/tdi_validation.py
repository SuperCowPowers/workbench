from __future__ import annotations

from typing import Iterable

import pandas as pd
from pathlib import Path

TDI_DATASET_SIZE = 750
TDI_ISOFORMS = ("CYP3A4", "CYP2D6")
TDI_VALUE_COLUMNS = tuple(f"{cyp}_is_TDI" for cyp in TDI_ISOFORMS)

_VALID_BOOL_STRINGS = {"true", "false", "1", "0", "1.0", "0.0"}


def _as_set(values: Iterable[str]) -> set[str]:
    return {str(v) for v in values}


def validate_tdi_submission(
    tdi_predictions_file: Path,
    expected_ids: set[str] | None = None,
    required_id_columns: tuple[str, ...] = ("SMILES", "Molecule_Name"),
    required_value_columns: tuple[str, ...] = TDI_VALUE_COLUMNS,
) -> tuple[bool, list[str]]:
    """Validate a CYP TDI classification submission file.

    The submission must contain the compound identifiers plus a
    ``{CYP}_is_TDI`` column per evaluated isoform (CYP3A4 and CYP2D6): the
    predicted boolean label for whether a compound shows time-dependent
    inhibition of that isoform.

    Args:
        tdi_predictions_file (Path): Path to the submission CSV file.
        expected_ids (set[str] | None): Expected 'Molecule_Name' values. If
            provided, the submission is checked for missing/extra IDs
            instead of a fixed row count.
        required_id_columns (tuple[str, ...]): Identifier columns that must
            be present and non-null.
        required_value_columns (tuple[str, ...]): Prediction columns that
            must be present and contain only boolean values (True/False,
            or 1/0).

    Returns:
        tuple[bool, list[str]]: Whether the submission is valid, and a list
            of validation error messages (empty if valid).

    """
    errors: list[str] = []

    path = Path(tdi_predictions_file)
    if not path.exists():
        return False, [f"File does not exist: {path}"]

    try:
        tdi_predictions = pd.read_csv(path)
    except Exception as exc:
        return False, [f"Error reading CSV file: {exc}"]

    required_columns = (*required_id_columns, *required_value_columns)
    missing_columns = [col for col in required_columns if col not in tdi_predictions.columns]
    if missing_columns:
        errors.append(f"Missing required column(s): {missing_columns}")
        return False, errors

    if tdi_predictions.empty:
        errors.append("Submission is empty.")
        return False, errors

    null_id_rows = tdi_predictions[list(required_id_columns)].isna().any(axis=1).sum()
    if null_id_rows:
        errors.append(f"Found {null_id_rows} row(s) with missing identifier values.")

    if "Molecule_Name" in tdi_predictions.columns:
        duplicate_ids = tdi_predictions["Molecule_Name"].duplicated().sum()
        if duplicate_ids:
            errors.append(f"Found {duplicate_ids} duplicated 'Molecule_Name' value(s).")

    for col in required_value_columns:
        raw = tdi_predictions[col]
        null_count = raw.isna().sum()
        if null_count:
            errors.append(f"Column '{col}' contains {null_count} missing value(s).")
            continue

        normalized = raw.astype(str).str.strip().str.lower()
        invalid_mask = ~normalized.isin(_VALID_BOOL_STRINGS)
        n_invalid = int(invalid_mask.sum())
        if n_invalid:
            errors.append(
                f"Column '{col}' contains {n_invalid} value(s) that are not valid booleans (True/False)."
            )

    submitted_ids = _as_set(tdi_predictions["Molecule_Name"])
    if expected_ids is not None:
        expected_ids = _as_set(expected_ids)
        missing = sorted(expected_ids - submitted_ids)
        extra = sorted(submitted_ids - expected_ids)
        if missing:
            errors.append(f"Missing {len(missing)} expected molecule(s): {missing[:20]}")
        if extra:
            errors.append(f"Found {len(extra)} unexpected molecule(s): {extra[:20]}")
    elif len(tdi_predictions) != TDI_DATASET_SIZE:
        errors.append(
            f"Submission contains {len(tdi_predictions)} rows, expected {TDI_DATASET_SIZE}."
        )

    return len(errors) == 0, errors
