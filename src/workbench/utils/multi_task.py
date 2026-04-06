"""Utilities for building multi-task DataFrames from single-task sources."""

import logging
import pandas as pd

log = logging.getLogger("workbench")


def combine_multi_task_data(
    dataframes: list[pd.DataFrame],
    target_columns: list[list[str]],
    id_column: str = "id",
) -> pd.DataFrame:
    """Combine single-task DataFrames into a multi-task DataFrame.

    Computes the shared feature columns across all DataFrames, subsets each to
    shared features + its targets, concatenates, and collapses rows by id_column
    (first non-NaN per column). This ensures molecules appearing in multiple
    sources get all their targets on a single row.

    Args:
        dataframes: List of DataFrames, each containing id_column, 'smiles', shared
            feature columns, and one or more target columns. Caller must pre-normalize
            id columns (rename, cast to consistent type) before calling.
        target_columns: Parallel list where target_columns[i] names the target column(s)
            in dataframes[i]. Every other column is treated as a shared feature.
        id_column: Column to use as the identifier. Defaults to 'id'.

    Returns:
        Combined DataFrame with shared features + all target columns. Rows from
        sources missing a target will have NaN for that target.

    Raises:
        ValueError: If inputs are invalid (length mismatch, missing columns, etc.).
    """
    # --- Input validation ---
    if len(dataframes) != len(target_columns):
        raise ValueError(
            f"dataframes ({len(dataframes)}) and target_columns ({len(target_columns)}) must have the same length"
        )
    if not dataframes:
        raise ValueError("dataframes must be non-empty")

    for i, (df, targets) in enumerate(zip(dataframes, target_columns)):
        if id_column not in df.columns:
            raise ValueError(f"DataFrame {i} missing id_column '{id_column}'")
        if "smiles" not in df.columns:
            raise ValueError(f"DataFrame {i} missing 'smiles' column")
        missing = [t for t in targets if t not in df.columns]
        if missing:
            raise ValueError(f"DataFrame {i} missing target columns: {missing}")

    all_targets = [t for targets in target_columns for t in targets]
    if len(all_targets) != len(set(all_targets)):
        dupes = [t for t in all_targets if all_targets.count(t) > 1]
        raise ValueError(f"Duplicate target column names across DataFrames: {set(dupes)}")

    # --- Step 1: Compute shared feature columns ---
    reserved = {id_column, "smiles"} | set(all_targets)
    feature_sets = []
    for i, (df, targets) in enumerate(zip(dataframes, target_columns)):
        feature_cols = set(df.columns) - reserved - set(targets)
        feature_sets.append(feature_cols)

    shared_features = feature_sets[0]
    for fs in feature_sets[1:]:
        shared_features &= fs

    for i, fs in enumerate(feature_sets):
        dropped = fs - shared_features
        if dropped:
            log.info(f"DataFrame {i}: dropping {len(dropped)} non-shared columns")

    shared_features = sorted(shared_features)
    id_cols = [id_column] if id_column == "smiles" else [id_column, "smiles"]
    keep_cols = id_cols + shared_features
    log.info(f"Shared feature columns: {len(shared_features)}")

    # --- Step 2: Subset each DataFrame to shared columns + its targets, then concat ---
    aligned_dfs = []
    for df, targets in zip(dataframes, target_columns):
        cols = [c for c in keep_cols if c in df.columns] + targets
        aligned_dfs.append(df[cols])

    result = pd.concat(aligned_dfs, ignore_index=True)

    # --- Step 3: Collapse rows by id_column ---
    n_before = len(result)
    dup_counts = result[id_column].value_counts()
    dup_ids = dup_counts[dup_counts > 1]
    # Use mean for numeric columns (averages replicate measurements),
    # first for non-numeric columns (smiles, strings, etc.)
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in result.columns if c not in numeric_cols and c != id_column]
    agg_dict = {c: "mean" for c in numeric_cols}
    agg_dict.update({c: "first" for c in non_numeric_cols})
    result = result.groupby(id_column, as_index=False).agg(agg_dict)
    log.info(f"Collapsing {n_before} rows -> {len(result)} " f"({len(dup_ids)} molecules appear in multiple sources)")

    # --- Step 3: Diagnostics ---
    input_support = {}
    for df, targets in zip(dataframes, target_columns):
        for t in targets:
            input_support[t] = df[t].notna().sum()

    log.info(f"Combined DataFrame: {len(result)} rows, {len(result.columns)} columns")
    log.info(f"Targets ({len(all_targets)}):")
    empty_targets = []
    for t in all_targets:
        out_count = result[t].notna().sum()
        in_count = input_support[t]
        delta = out_count - in_count
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        suffix = f" — {abs(delta)} dup ids, targets averaged" if delta < 0 else ""
        log.info(f"  {t}: {in_count} input -> {out_count} output ({delta_str}){suffix}")
        if out_count == 0:
            empty_targets.append(t)

    if empty_targets:
        log.warning(f"WARNING: Targets with ALL NaN values (no data): {empty_targets}")

    # Target coverage patterns
    target_pattern = result[all_targets].notna()
    pattern_labels = target_pattern.apply(
        lambda row: " + ".join(t for t, present in zip(all_targets, row) if present) or "(none)",
        axis=1,
    )
    log.info("Target coverage patterns:")
    for pattern, count in pattern_labels.value_counts().items():
        log.info(f"  {count:>6}  {pattern}")

    return result


def validate_multi_task_data(
    df: pd.DataFrame,
    target_columns: list[str],
    id_column: str = "id",
) -> None:
    """Validate a multi-task DataFrame before model training or ingestion.

    Checks for common data issues: null/duplicate IDs, missing smiles, empty targets,
    fully-NaN feature columns, and featureless rows.

    Args:
        df: Multi-task DataFrame to validate.
        target_columns: List of active target column names.
        id_column: ID column name.

    Raises:
        ValueError: If any critical validation check fails.
    """
    errors = []
    warnings = []

    # 1. Check id column has no NaN
    n_null_id = df[id_column].isna().sum()
    if n_null_id > 0:
        errors.append(f"{id_column} has {n_null_id} NaN values")

    # 2. Check for duplicate IDs
    n_dup_id = df[id_column].duplicated().sum()
    if n_dup_id > 0:
        errors.append(f"{id_column} has {n_dup_id} duplicate values")

    # 3. Check smiles column exists and has no NaN
    if "smiles" not in df.columns:
        errors.append("'smiles' column missing from DataFrame")
    else:
        n_null_smiles = df["smiles"].isna().sum()
        if n_null_smiles > 0:
            errors.append(f"'smiles' has {n_null_smiles} NaN values")

        n_dup_smiles = df["smiles"].duplicated().sum()
        if n_dup_smiles > 0:
            warnings.append(f"'smiles' has {n_dup_smiles} duplicate values (may be expected)")

    # 4. Check all active targets are present and non-empty
    for t in target_columns:
        if t not in df.columns:
            errors.append(f"Target '{t}' missing from DataFrame")
        elif df[t].notna().sum() == 0:
            errors.append(f"Target '{t}' has zero non-null values")

    # 5. Check feature columns for unexpected NaN patterns
    feature_cols = [c for c in df.columns if c not in [id_column, "smiles"] + list(target_columns)]
    if feature_cols:
        feature_null_frac = df[feature_cols].isna().mean()
        fully_null_features = feature_null_frac[feature_null_frac == 1.0]
        if len(fully_null_features) > 0:
            warnings.append(
                f"{len(fully_null_features)} feature columns are entirely NaN: "
                f"{list(fully_null_features.index[:5])}..."
            )

        # Check for featureless rows (e.g. from smiles-only merges)
        high_null_rows = df[feature_cols].isna().all(axis=1)
        n_featureless = high_null_rows.sum()
        if n_featureless > 0:
            warnings.append(f"{n_featureless} rows have no feature values (likely from smiles-only merge)")

    # Report
    for w in warnings:
        log.warning(f"  {w}")
    for e in errors:
        log.error(f"  {e}")

    if errors:
        raise ValueError(f"Validation failed with {len(errors)} error(s):\n" + "\n".join(f"  - {e}" for e in errors))

    log.info(f"  Validated: {len(df)} rows, {len(target_columns)} targets, {len(feature_cols)} features")
