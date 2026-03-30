"""Alignment utilities for validating pulled ADMET datasets.

Provides generic sanity checks that apply to any assay type:
  - Value range validation against expected bounds
  - Distribution shape analysis (variance, skew, kurtosis)
  - Per-source scale consistency
  - Cross-source agreement for overlapping compounds
  - Basic data quality (missing values, duplicates)

Usage:
    from alignment_utils import run_alignment_checks

    # After merging and deduplicating
    run_alignment_checks(
        df=merged_df,
        value_col="logp_mean",
        assay_name="LogP",
        expected_range=(-5, 10),
        expected_mean=(1, 3),
    )
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)


def run_alignment_checks(
    df: pd.DataFrame,
    value_col: str = "logp_mean",
    std_col: str | None = "logp_std",
    count_col: str | None = "logp_count",
    sources_col: str = "sources",
    assay_name: str = "LogP",
    expected_range: tuple[float, float] = (-5, 10),
    expected_mean: tuple[float, float] = (0, 5),
    expected_std: tuple[float, float] = (0.5, 4.0),
    outlier_pct_threshold: float = 5.0,
    cross_source_std_warn: float = 1.0,
) -> list[str]:
    """Run alignment checks on a merged/deduplicated assay dataset.

    Args:
        df: Merged DataFrame (one row per unique compound).
        value_col: Column with the mean assay value.
        std_col: Column with cross-source std (None to skip cross-source checks).
        count_col: Column with per-compound source count (None to skip).
        sources_col: Column with pipe-delimited source names.
        assay_name: Human-readable assay name for output.
        expected_range: (low, high) bounds for reasonable values.
        expected_mean: (low, high) expected range for the dataset mean.
        expected_std: (low, high) expected range for the dataset std dev.
        outlier_pct_threshold: Warn if more than this % of values are outside expected_range.
        cross_source_std_warn: Warn if mean cross-source std exceeds this.

    Returns:
        List of issue strings (empty if all checks pass).
    """
    print("\n" + "=" * 60)
    print(f"Alignment Checks: {assay_name}")
    print("=" * 60)

    issues: list[str] = []
    vals = df[value_col]
    lo, hi = expected_range

    # --- 1. Value range ---
    n_below = (vals < lo).sum()
    n_above = (vals > hi).sum()
    n_extreme = n_below + n_above
    pct_extreme = 100 * n_extreme / len(df) if len(df) > 0 else 0

    print(f"\n1. Value Range (expected [{lo}, {hi}])")
    print(f"   Min: {vals.min():.2f}  Max: {vals.max():.2f}  Mean: {vals.mean():.2f}  Median: {vals.median():.2f}")
    print(f"   Outside [{lo}, {hi}]: {n_extreme} ({pct_extreme:.1f}%)")
    if pct_extreme > outlier_pct_threshold:
        msg = f"WARNING: {pct_extreme:.1f}% of values outside expected [{lo}, {hi}] range"
        issues.append(msg)
        print(f"   ** {msg}")
    else:
        print("   OK")

    # --- 2. Distribution shape ---
    data_std = vals.std()
    skew = vals.skew()
    kurt = vals.kurtosis()

    print("\n2. Distribution Shape")
    print(f"   Std: {data_std:.2f}  Skew: {skew:.2f}  Kurtosis: {kurt:.2f}")

    std_lo, std_hi = expected_std
    if data_std < std_lo:
        msg = f"WARNING: very low variance (std={data_std:.2f}) — possible unit or scale issue"
        issues.append(msg)
        print(f"   ** {msg}")
    elif data_std > std_hi:
        msg = f"WARNING: very high variance (std={data_std:.2f}) — possible mixed scales"
        issues.append(msg)
        print(f"   ** {msg}")
    else:
        print("   OK")

    mean_lo, mean_hi = expected_mean
    if not (mean_lo <= vals.mean() <= mean_hi):
        msg = f"WARNING: dataset mean ({vals.mean():.2f}) outside expected [{mean_lo}, {mean_hi}]"
        issues.append(msg)
        print(f"   ** {msg}")

    # Text histogram
    _print_histogram(vals, lo, hi)

    # --- 3. Per-source scale check ---
    if sources_col in df.columns:
        print("\n3. Per-Source Scale Consistency")
        sources = df[sources_col].str.split("|").explode().unique()
        overall_mean = vals.mean()
        overall_std = vals.std()

        for src in sorted(sources):
            mask = df[sources_col].str.contains(src, regex=False)
            src_vals = df.loc[mask, value_col]
            print(f"   {src:<35s} n={len(src_vals):>6,}  " f"mean={src_vals.mean():>6.2f}  std={src_vals.std():.2f}")

            if abs(src_vals.mean() - overall_mean) > 2 * overall_std:
                msg = (
                    f"WARNING: {src} mean ({src_vals.mean():.2f}) deviates "
                    f"significantly from overall ({overall_mean:.2f})"
                )
                issues.append(msg)
                print(f"   ** {msg}")

        if not any("scale" in i.lower() or "deviates" in i.lower() for i in issues):
            print("   OK — sources are on consistent scales")

    # --- 4. Cross-source agreement ---
    if count_col and count_col in df.columns and std_col and std_col in df.columns:
        multi_source = df[df[count_col] > 1]
        print("\n4. Cross-Source Agreement")
        print(f"   Compounds in multiple sources: {len(multi_source):,}")
        if len(multi_source) > 0:
            mean_xstd = multi_source[std_col].mean()
            median_xstd = multi_source[std_col].median()
            max_xstd = multi_source[std_col].max()
            n_high = (multi_source[std_col] > cross_source_std_warn).sum()

            print(f"   Cross-source std — mean: {mean_xstd:.2f}  median: {median_xstd:.2f}  max: {max_xstd:.2f}")
            print(f"   High disagreement (std > {cross_source_std_warn}): {n_high}")
            if mean_xstd > cross_source_std_warn:
                msg = f"WARNING: high mean cross-source disagreement (std={mean_xstd:.2f})"
                issues.append(msg)
                print(f"   ** {msg}")
            else:
                print("   OK")
        else:
            print("   (no overlapping compounds to compare)")

    # --- 5. Data quality ---
    print("\n5. Data Quality")
    print(f"   Total unique compounds: {len(df):,}")
    smiles_col = "smiles" if "smiles" in df.columns else "canon_smiles"
    n_nan_smiles = df[smiles_col].isna().sum() if smiles_col in df.columns else 0
    n_nan_val = df[value_col].isna().sum()
    print(f"   Missing SMILES: {n_nan_smiles}")
    print(f"   Missing {assay_name}: {n_nan_val}")
    if n_nan_smiles > 0 or n_nan_val > 0:
        msg = f"WARNING: {n_nan_smiles} missing SMILES, {n_nan_val} missing {assay_name}"
        issues.append(msg)

    # --- Summary ---
    print("\n" + "-" * 60)
    if issues:
        print(f"ALIGNMENT ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("ALL CHECKS PASSED")
    print("=" * 60)

    return issues


def _print_histogram(vals: pd.Series, expected_lo: float, expected_hi: float):
    """Print a text-based histogram of the value distribution."""
    # Build bins: one below expected, several within, one above
    full_lo = min(vals.min(), expected_lo - 5)
    full_hi = max(vals.max(), expected_hi + 10)

    bins = []
    # Below expected range
    if full_lo < expected_lo:
        bins.append((full_lo, expected_lo))
    # Within expected range — subdivide
    span = expected_hi - expected_lo
    if span > 0:
        n_inner = min(8, max(3, int(span)))
        step = span / n_inner
        for i in range(n_inner):
            bins.append((expected_lo + i * step, expected_lo + (i + 1) * step))
    # Above expected range
    if full_hi > expected_hi:
        bins.append((expected_hi, full_hi))

    max_bar = 40
    counts = [(lo, hi, ((vals >= lo) & (vals < hi)).sum()) for lo, hi in bins]
    max_count = max(c for _, _, c in counts) if counts else 1

    print("\n   Distribution:")
    for lo, hi, count in counts:
        bar = "#" * int(max_bar * count / max_count) if max_count > 0 else ""
        print(f"   [{lo:>6.1f},{hi:>6.1f}) {count:>6,} {bar}")
