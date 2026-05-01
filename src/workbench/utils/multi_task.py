"""Utilities for building multi-task DataFrames from single-task sources."""

import logging
from functools import reduce

import numpy as np
import pandas as pd

log = logging.getLogger("workbench")


def compute_inverse_count_task_weights(targets: np.ndarray) -> np.ndarray:
    """Per-task loss weights inversely proportional to non-NaN row counts, mean-normalized to 1.

    Use to equalize each task's gradient contribution when target columns have
    unequal coverage in a multi-task model.

    Only appropriate for *symmetric* multi-task setups where every task is an end
    product. For *primary + auxiliary* setups (only one task's predictions are
    used downstream), this is the wrong knob — it up-weights smaller auxiliaries
    and starves the primary. Use manual primary-favored weights instead, e.g.
    [1.0, 0.3] with the primary first.

    Args:
        targets: (n_rows, n_tasks) float array of target values; NaN means missing.

    Returns:
        (n_tasks,) float32 array of weights, mean-normalized to 1.

    Raises:
        ValueError: If any task has zero non-NaN rows.
    """
    if targets.ndim != 2:
        raise ValueError(f"targets must be 2D (n_rows, n_tasks), got shape {targets.shape}")

    counts = np.array([np.sum(~np.isnan(targets[:, t])) for t in range(targets.shape[1])], dtype=np.float32)
    if not (counts > 0).all():
        raise ValueError(f"All tasks must have at least one non-NaN row; got counts {counts.tolist()}")

    inv = 1.0 / counts
    return (inv / inv.mean()).astype(np.float32)


def combine_multi_task_data(
    dataframes: list[pd.DataFrame],
    target_columns: list[list[str]],
    id_column: str = "id",
    merge_on_smiles: bool = False,
) -> pd.DataFrame:
    """Combine single-task DataFrames into a multi-task DataFrame.

    Computes the shared feature columns across all DataFrames, subsets each to
    shared features + its targets, concatenates, and collapses rows by the merge
    key (first non-NaN per column). This ensures molecules appearing in multiple
    sources get all their targets on a single row.

    Convention: the first DataFrame's target(s) are treated as the *primary* task
    downstream (e.g. by `assess_multi_task_data` and chemprop `task_weights`).

    Args:
        dataframes: List of DataFrames, primary task first. Each contains id_column,
            'smiles', shared feature columns, and one or more target columns. Caller
            must pre-normalize id columns (rename, cast to consistent type) first.
        target_columns: Parallel list where target_columns[i] names the target column(s)
            in dataframes[i]. Every other column is treated as a shared feature.
        id_column: Column to use as the identifier. Defaults to 'id'.
        merge_on_smiles: If True, collapse rows by 'smiles' instead of id_column.
            Use when combining external/public data where IDs have no correspondence
            to internal IDs. Defaults to False.

    Returns:
        Combined DataFrame with shared features + all target columns. Rows from
        sources missing a target will have NaN for that target.

    Raises:
        ValueError: If inputs are invalid (length mismatch, missing columns, etc.)
            or if any target column ends up with all NaN values.
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

    merge_key = "smiles" if merge_on_smiles else id_column

    # --- Drop rows with NaN smiles ---
    for i, df in enumerate(dataframes):
        n_null = df["smiles"].isna().sum()
        if n_null > 0:
            log.warning(f"DataFrame {i}: dropping {n_null} rows with NaN smiles")
            dataframes[i] = df.dropna(subset=["smiles"])

    # --- Step 1: Compute shared feature columns ---
    reserved = {id_column, "smiles"} | set(all_targets)
    all_feature_sets = [set(df.columns) - reserved - set(t) for df, t in zip(dataframes, target_columns)]
    shared_features = sorted(reduce(set.intersection, all_feature_sets))
    for i, fs in enumerate(all_feature_sets):
        dropped = len(fs) - len(shared_features)
        if dropped:
            log.info(f"DataFrame {i}: dropping {dropped} non-shared columns")

    keep_cols = [id_column, "smiles"] + shared_features
    log.info(f"Shared feature columns: {len(shared_features)}")

    # --- Step 2: Subset each DataFrame to shared columns + its targets, then concat ---
    col_dtypes = {}
    for df in dataframes:
        for col in df.columns:
            if col not in col_dtypes and not df[col].isna().all():
                col_dtypes[col] = df[col].dtype

    aligned_dfs = []
    for df, targets in zip(dataframes, target_columns):
        sub = df[keep_cols + targets].copy()
        # Pre-add missing target columns to avoid FutureWarning on concat
        for t in all_targets:
            if t not in sub.columns:
                sub[t] = pd.array([pd.NA] * len(sub), dtype="Float64")
        # Cast all-NA columns to match dtype from DataFrames that have data
        for col in sub.columns:
            if sub[col].isna().all() and col in col_dtypes:
                sub[col] = sub[col].astype(col_dtypes[col])
        aligned_dfs.append(sub)

    result = pd.concat(aligned_dfs, ignore_index=True)

    # --- Step 3: Collapse rows by merge key ---
    n_before = len(result)
    dup_counts = result[merge_key].value_counts()
    dup_ids = dup_counts[dup_counts > 1]
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in result.columns if c not in numeric_cols and c != merge_key]
    agg_dict = {c: "mean" for c in numeric_cols}
    agg_dict.update({c: "first" for c in non_numeric_cols})
    result = result.groupby(merge_key, as_index=False).agg(agg_dict)
    log.info(f"Collapsing {n_before} rows -> {len(result)} ({len(dup_ids)} molecules appear in multiple sources)")

    # --- Step 4: Diagnostics ---
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
        raise ValueError(f"Targets with ALL NaN values (no data): {empty_targets}")

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


def pull_multi_task_data(
    id_based_sources: dict,
    smiles_based_sources: dict = None,
    id_column: str = "id",
) -> pd.DataFrame:
    """Pull and combine multiple FeatureSets into a multi-task DataFrame.

    Two-pass merge convention:
        1. ID-based sources are combined via outer join on `id_column`.
        2. SMILES-based sources are joined onto the result via canonical SMILES
           (use this for external/public data with no shared id namespace).

    Each source config is a dict shaped like:
        {
            "target_info": {<src_col>: <output_col>, ...}  # required; column rename map
            "src_id_col":  "<non_default_id_column>"       # optional; if the FS uses a different id column
        }

    Or as a shortcut, `target_info` may be a list of column names that already
    match the desired output names (no rename needed).

    Args:
        id_based_sources: Mapping from FeatureSet name to source config.
            Primary task convention: the first entry is treated as primary by
            downstream consumers (e.g. assess_multi_task_data, chemprop task_weights).
        smiles_based_sources: Optional mapping (same shape) for sources joined
            on canonical SMILES rather than id_column. Defaults to None / empty.
        id_column: Canonical id column name across the id-based sources.
            Defaults to "id".

    Returns:
        Combined multi-task DataFrame from `combine_multi_task_data`, with all
        target columns aligned and shared features intersected.
    """
    # Local import: workbench.api is heavy and not needed at module import time.
    from workbench.api import FeatureSet

    smiles_based_sources = smiles_based_sources or {}

    def _pull_and_normalize(fs_name: str, fs_config: dict) -> tuple[pd.DataFrame, list[str]]:
        target_info = fs_config["target_info"]
        src_id = fs_config.get("src_id_col", id_column)

        df = FeatureSet(fs_name).pull_dataframe()

        # Normalize id column
        if src_id != id_column:
            df = df.rename(columns={src_id: id_column})
        df[id_column] = df[id_column].astype(str)

        # Resolve target columns (rename if dict, else assume already-named)
        if isinstance(target_info, dict):
            df = df.rename(columns=target_info)
            target_cols = list(target_info.values())
        else:
            target_cols = list(target_info)

        log.info(f"  {fs_name}: {len(df):,} rows, {len(df.columns)} cols")
        return df, target_cols

    # Pass 1: id-based outer join
    id_dfs, id_targets = [], []
    for fs_name, fs_config in id_based_sources.items():
        df, target_cols = _pull_and_normalize(fs_name, fs_config)
        id_dfs.append(df)
        id_targets.append(target_cols)
    merged = combine_multi_task_data(id_dfs, id_targets, id_column=id_column)

    # Pass 2: smiles-based join (e.g. external/public data)
    merged_targets = [t for tl in id_targets for t in tl]
    for fs_name, fs_config in smiles_based_sources.items():
        df, target_cols = _pull_and_normalize(fs_name, fs_config)
        merged = combine_multi_task_data([merged, df], [merged_targets, target_cols], merge_on_smiles=True)
        merged_targets.extend(target_cols)

    return merged


def assess_multi_task_data(
    df: pd.DataFrame,
    target_columns: list[str],
    extension_ratio_threshold: float = 0.5,
    min_n_shared: int = 10,
) -> pd.DataFrame:
    """Pre-flight check: score each auxiliary along two independent axes.

    Lightweight, label-only diagnostic on a pre-combined multi-task DataFrame.
    For a deeper cross-chemistry concordance analysis (fingerprint-based
    neighbor residuals, similarity coverage, UMAP), see
    `workbench.algorithms.dataframe.dataset_comparison.DatasetComparison`.

    Multi-task data has two distinct regions, with very different lift dynamics:

    OVERLAP region (compounds where both primary and aux have ground truth)
        Two heads predict from the same encoder embedding. Their label
        relationship determines the effect on the shared encoder:

          r in [0.4, 0.95]  -> Beneficial : sweet spot — heads predict related
                              but distinct targets, forcing richer encoder
                              representations (classic transfer-learning win)
          r > 0.95          -> Neutral    : labels redundant; the aux head's
                              gradient just re-weights primary supervision —
                              no harm, no new information
          r < 0.4           -> Harmful    : heads disagree; encoder gradients
                              conflict — negative-transfer risk
          n_shared too low  -> N/A        : not enough overlap to measure

    EXTENSION region (aux-only compounds; primary head is masked here)
        Primary head sees no gradient on these rows, so the worst case is
        "encoder learns useless features" — extension cannot actively hurt.
        Contribution scales with two factors:
            volume     = n_aux_only / n_primary
            similarity = how related the aux task is to primary (estimated
                         from r on overlap, or NaN when no overlap to measure)

          Strong   - substantial volume AND similar task (or unmeasured)
          Modest   - either limited volume OR weak similarity
          Minimal  - low volume and weak similarity
          None     - no aux-only compounds

    The two scores combine into an actionable RECOMMENDATION:
          Use      - either axis is genuinely lifting; no harm signals
          Marginal - limited contribution; consider domain knowledge
          Risky    - harmful overlap that *might* be rescued by extension
          Skip     - no useful contribution

    Args:
        df: Combined multi-task DataFrame (typically from combine_multi_task_data).
        target_columns: Target column names. The first is treated as the primary;
            subsequent entries are auxiliaries assessed against the primary.
        extension_ratio_threshold: n_aux_only / n_primary above which the
            extension region is considered to have substantial volume. Default 0.5.
        min_n_shared: minimum shared compounds required before the overlap-region
            Pearson r is trusted. Default 10.

    Returns:
        DataFrame with one row per auxiliary, columns:
            auxiliary, n_primary, n_aux, n_shared, n_aux_only, pearson_r,
            r_confidence, overlap, extension, recommendation, value_range, reason.

        r_confidence reflects how trustworthy the pearson_r value is:
            high       (n_shared >= 100)
            moderate   (n_shared >= 30)
            low        (n_shared >= min_n_shared but < 30)
            unmeasured (n_shared < min_n_shared)

        value_range is a sanity check on whether primary and aux value
        distributions are roughly compatible (their IQRs overlap):
            ok    (IQRs overlap; ranges look compatible)
            WARN  (IQRs do not overlap; possible column mismatch or scale issue)
    """
    if len(target_columns) < 2:
        raise ValueError(f"Need at least 2 targets (primary + 1 auxiliary), got {len(target_columns)}")
    missing = [t for t in target_columns if t not in df.columns]
    if missing:
        raise ValueError(f"Target columns missing from DataFrame: {missing}")

    primary = target_columns[0]
    primary_mask = df[primary].notna()
    n_primary = int(primary_mask.sum())
    if n_primary == 0:
        raise ValueError(f"Primary target '{primary}' has zero non-null values")

    log.info(f"Multi-task assessment (primary = '{primary}', n = {n_primary:,})")
    rows = []
    for aux in target_columns[1:]:
        aux_mask = df[aux].notna()
        n_aux = int(aux_mask.sum())
        n_shared = int((primary_mask & aux_mask).sum())
        n_aux_only = int((~primary_mask & aux_mask).sum())

        # Pearson on shared compounds (need enough overlap to be meaningful)
        if n_shared >= min_n_shared:
            r = float(df.loc[primary_mask & aux_mask, [primary, aux]].corr().iloc[0, 1])
        else:
            r = float("nan")

        overlap, overlap_reason = _assess_overlap(r, n_shared, min_n_shared)
        extension, extension_reason = _assess_extension(r, n_aux_only, n_primary, extension_ratio_threshold)
        recommendation, rec_reason = _combine_assessments(overlap, extension)
        r_confidence = _confidence_tier(n_shared, min_n_shared)
        value_range, value_range_reason = _value_range_check(df[primary], df[aux])
        reason = f"overlap: {overlap_reason}. extension: {extension_reason}. -> {rec_reason}"

        rows.append(
            {
                "auxiliary": aux,
                "n_primary": n_primary,
                "n_aux": n_aux,
                "n_shared": n_shared,
                "n_aux_only": n_aux_only,
                "pearson_r": r,
                "r_confidence": r_confidence,
                "overlap": overlap,
                "extension": extension,
                "recommendation": recommendation,
                "value_range": value_range,
                "reason": reason,
            }
        )
        r_str = f"r={r:.3f}" if not np.isnan(r) else "r=NA"
        log.info(
            f"  {aux}: shared={n_shared:,} aux_only={n_aux_only:,} {r_str} ({r_confidence}) "
            f"-> overlap={overlap}, extension={extension} -> {recommendation}"
        )
        if value_range == "WARN":
            log.warning(f"  {aux}: value-range WARN — {value_range_reason}")

    return pd.DataFrame(rows)


def _confidence_tier(n_shared: int, min_n_shared: int) -> str:
    """Bucket how trustworthy a Pearson r is, given the shared-compound count."""
    if n_shared < min_n_shared:
        return "unmeasured"
    if n_shared < 30:
        return "low"
    if n_shared < 100:
        return "moderate"
    return "high"


def _value_range_check(primary_vals: pd.Series, aux_vals: pd.Series) -> tuple[str, str]:
    """Sanity check that primary/aux value distributions look compatible.

    Compares IQR overlap. WARN when the middle 50% of each distribution does
    not intersect — usually indicates a column mismatch (e.g. accidentally
    mixing a percent column with a log-scale column) or an unexpected
    unit/scale difference between sources.
    """
    p = primary_vals.dropna()
    a = aux_vals.dropna()
    if len(p) < 5 or len(a) < 5:
        return ("ok", "too few values to check")

    p_med, p_q1, p_q3 = float(p.median()), float(p.quantile(0.25)), float(p.quantile(0.75))
    a_med, a_q1, a_q3 = float(a.median()), float(a.quantile(0.25)), float(a.quantile(0.75))
    summary = (
        f"primary median={p_med:.2f} IQR=[{p_q1:.2f}, {p_q3:.2f}]; "
        f"aux median={a_med:.2f} IQR=[{a_q1:.2f}, {a_q3:.2f}]"
    )

    iqr_overlap = (p_q3 >= a_q1) and (a_q3 >= p_q1)
    if iqr_overlap:
        return ("ok", summary)
    return ("WARN", f"{summary} — IQRs do not overlap, possible column or scale mismatch")


def _assess_overlap(r: float, n_shared: int, min_n_shared: int) -> tuple[str, str]:
    """Score the overlap region. See assess_multi_task_data() for thresholds."""
    if n_shared < min_n_shared:
        return ("N/A", f"only {n_shared} shared compounds (need >= {min_n_shared} to score)")
    if 0.4 <= r <= 0.95:
        return ("Beneficial", f"sweet-spot r={r:.2f} on {n_shared:,} shared compounds — encoder learns richer features")
    if r > 0.95:
        return ("Neutral", f"redundant r={r:.2f} on {n_shared:,} shared compounds — aux head just re-weights primary")
    return (
        "Harmful",
        f"discordant r={r:.2f} on {n_shared:,} shared compounds — gradient conflict, negative-transfer risk",
    )


def _assess_extension(
    r: float,
    n_aux_only: int,
    n_primary: int,
    ratio_threshold: float,
) -> tuple[str, str]:
    """Score the extension region. See assess_multi_task_data() for thresholds."""
    if n_aux_only == 0:
        return ("None", "no aux-only compounds")

    ratio = n_aux_only / n_primary if n_primary > 0 else 0.0
    has_volume = ratio >= ratio_threshold

    # Task similarity: estimated from r when overlap is measurable; otherwise unknown.
    if np.isnan(r):
        sim_str = "task similarity unknown (no overlap to measure)"
        # Optimistic when no overlap: the user presumably chose related tasks
        if has_volume:
            return ("Strong", f"{ratio:.1f}x primary of novel chemistry; {sim_str}")
        return ("Modest", f"{n_aux_only:,} aux-only compounds ({ratio:.1f}x primary); {sim_str}")

    similar = r >= 0.4
    if has_volume and similar:
        return ("Strong", f"{ratio:.1f}x primary of novel chemistry x similar task (r={r:.2f})")
    if has_volume:
        return ("Modest", f"{ratio:.1f}x primary of novel chemistry but weak similarity (r={r:.2f})")
    if similar:
        return ("Modest", f"limited volume ({ratio:.1f}x primary) but similar task (r={r:.2f})")
    return ("Minimal", f"low volume ({ratio:.1f}x primary) and weak similarity (r={r:.2f})")


def _combine_assessments(overlap: str, extension: str) -> tuple[str, str]:
    """Combine overlap + extension scores into an actionable recommendation."""
    if overlap == "Harmful":
        if extension in ("Strong", "Modest"):
            return ("Risky", "harmful overlap; extension might rescue but cross-task signal could hurt primary")
        return ("Skip", "negative transfer from overlap with no extension to compensate")

    if overlap == "Beneficial":
        if extension in ("Strong", "Modest"):
            return ("Use", "both mechanisms contribute lift")
        return ("Use", "cross-task signal contributes lift")

    # overlap is Neutral or N/A — extension is the only available mechanism
    if extension == "Strong":
        return ("Use", "extension is the primary lift mechanism")
    if extension == "Modest":
        return ("Marginal", "limited extension lift; consider domain knowledge")
    return ("Skip", "no clear lift mechanism")


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


if __name__ == "__main__":
    """Exercise the multi-task utility methods with small synthetic datasets."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)
    rng = np.random.default_rng(42)

    # =====================================================================
    # 1. compute_inverse_count_task_weights
    # =====================================================================
    print("\n=== compute_inverse_count_task_weights ===")
    targets_arr = np.array(
        [
            [1.0, np.nan, np.nan],
            [2.0, 5.0, np.nan],
            [np.nan, 6.0, np.nan],
            [3.0, 7.0, 9.0],
            [np.nan, np.nan, 10.0],
        ]
    )
    weights = compute_inverse_count_task_weights(targets_arr)
    print("  per-task non-NaN counts: [3, 3, 2]")
    print(f"  inverse-count weights (mean-normalized to 1): {weights.tolist()}")

    # =====================================================================
    # 2. combine_multi_task_data — id-based merge with partial overlap
    # =====================================================================
    print("\n=== combine_multi_task_data (id merge) ===")
    df_p = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "smiles": ["CC", "CCC", "CCCC", "CCCCC"],
            "feat_a": [0.1, 0.2, 0.3, 0.4],
            "primary": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df_a = pd.DataFrame(
        {
            "id": [3, 4, 5, 6],
            "smiles": ["CCCC", "CCCCC", "C1CC1", "C1CCC1"],
            "feat_a": [0.3, 0.4, 0.5, 0.6],
            "aux": [33.0, 44.0, 55.0, 66.0],
        }
    )
    combined = combine_multi_task_data([df_p, df_a], [["primary"], ["aux"]], id_column="id")
    print(combined.sort_values("id").to_string(index=False))

    # =====================================================================
    # 3. assess_multi_task_data — exercise each verdict path
    # =====================================================================
    def make_mt_df(
        n_shared: int, n_primary_only: int, n_aux_only: int, signal: float = 1.0, noise: float = 4.0
    ) -> pd.DataFrame:
        """Build a synthetic MT df with the requested overlap structure.

        signal/noise tune label correlation on shared rows: aux = signal*primary + N(0, noise).
        Empirical r ~ signal*sigma_x / sqrt(signal^2*sigma_x^2 + noise^2), with sigma_x=5.
        """
        primary_vals = rng.standard_normal(n_shared + n_primary_only) * 5
        aux_shared = signal * primary_vals[:n_shared] + rng.standard_normal(n_shared) * noise
        aux_only_vals = rng.standard_normal(n_aux_only) * 5
        return pd.DataFrame(
            {
                "primary": np.concatenate([primary_vals, [np.nan] * n_aux_only]),
                "aux": np.concatenate([aux_shared, [np.nan] * n_primary_only, aux_only_vals]),
            }
        )

    # Each case checks (overlap, extension, recommendation, r_confidence) as a tuple.
    cases = [
        # (label, df, expected_overlap, expected_extension, expected_recommendation, expected_r_confidence)
        (
            "sweet-spot + extended",
            make_mt_df(100, 100, 800, signal=1.0, noise=4.0),
            "Beneficial",
            "Strong",
            "Use",
            "high",
        ),
        (
            "sweet-spot, no extended",
            make_mt_df(100, 100, 0, signal=1.0, noise=4.0),
            "Beneficial",
            "None",
            "Use",
            "high",
        ),
        ("redundant + extended", make_mt_df(100, 100, 800, signal=1.0, noise=0.5), "Neutral", "Strong", "Use", "high"),
        ("redundant, no extended", make_mt_df(100, 100, 0, signal=1.0, noise=0.5), "Neutral", "None", "Skip", "high"),
        (
            "discordant + extended",
            make_mt_df(100, 100, 800, signal=0.1, noise=5.0),
            "Harmful",
            "Modest",
            "Risky",
            "high",
        ),
        ("discordant, no extended", make_mt_df(100, 100, 0, signal=0.1, noise=5.0), "Harmful", "None", "Skip", "high"),
        (
            "discordant + low-volume",
            make_mt_df(100, 100, 20, signal=0.1, noise=5.0),
            "Harmful",
            "Minimal",
            "Skip",
            "high",
        ),
        (
            "zero overlap, extended",
            make_mt_df(0, 100, 500, signal=1.0, noise=0.5),
            "N/A",
            "Strong",
            "Use",
            "unmeasured",
        ),
        (
            "redundant + low-volume",
            make_mt_df(100, 100, 20, signal=1.0, noise=0.5),
            "Neutral",
            "Modest",
            "Marginal",
            "high",
        ),
        # r_confidence tiers driven by n_shared
        (
            "sweet-spot, n_shared=15 (low confidence)",
            make_mt_df(15, 100, 200, signal=1.0, noise=4.0),
            "Beneficial",
            "Strong",
            "Use",
            "low",
        ),
        (
            "sweet-spot, n_shared=50 (moderate)",
            make_mt_df(50, 100, 200, signal=1.0, noise=4.0),
            "Beneficial",
            "Strong",
            "Use",
            "moderate",
        ),
    ]
    print("\n=== assess_multi_task_data ===")
    for label, df, exp_overlap, exp_ext, exp_rec, exp_conf in cases:
        print(f"\n--- {label} ---")
        result = assess_multi_task_data(df, ["primary", "aux"])
        print(result.drop(columns=["reason"]).to_string(index=False))
        row = result.iloc[0]
        assert row["overlap"] == exp_overlap, f"  overlap MISMATCH: expected {exp_overlap}, got {row['overlap']}"
        assert row["extension"] == exp_ext, f"  extension MISMATCH: expected {exp_ext}, got {row['extension']}"
        assert (
            row["recommendation"] == exp_rec
        ), f"  recommendation MISMATCH: expected {exp_rec}, got {row['recommendation']}"
        assert (
            row["r_confidence"] == exp_conf
        ), f"  r_confidence MISMATCH: expected {exp_conf}, got {row['r_confidence']}"
        assert row["value_range"] == "ok", f"  value_range MISMATCH: expected ok, got {row['value_range']}"

    # value_range WARN: aux on a totally different scale than primary
    print("\n--- value-range WARN (aux on a different scale) ---")
    df_scale = make_mt_df(100, 100, 200, signal=1.0, noise=4.0)
    df_scale["aux"] = df_scale["aux"] * 100 + 1000  # shift aux into a non-overlapping range
    result_scale = assess_multi_task_data(df_scale, ["primary", "aux"])
    print(result_scale.drop(columns=["reason"]).to_string(index=False))
    assert result_scale.iloc[0]["value_range"] == "WARN", "expected value_range=WARN for shifted aux"

    # The combined frame from step 2 has n_shared=2 (< min_n_shared=10) -> overlap='N/A'
    print("\n--- assess on combined df from step 2 (small n_shared) ---")
    print(assess_multi_task_data(combined, ["primary", "aux"]).drop(columns=["reason"]).to_string(index=False))

    # =====================================================================
    # 4. validate_multi_task_data — clean pass on the combined frame
    # =====================================================================
    print("\n=== validate_multi_task_data ===")
    validate_multi_task_data(combined, ["primary", "aux"], id_column="id")

    print("\nAll tests complete.")
