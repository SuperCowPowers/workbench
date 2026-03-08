"""Dataset Alignment plots and summary methods (culled from DatasetAlignment class).

These visualization and summary methods were extracted during a refactor to focus
DatasetAlignment on backing a scatter+contour UI. They are preserved here for
potential future use.

All functions take the relevant DatasetAlignment data as arguments rather than
depending on the class instance, making them reusable.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("workbench")


# =============================================================================
# Convenience Query Methods
# =============================================================================


def summary_stats(overlap_df: pd.DataFrame) -> pd.DataFrame:
    """Return distribution statistics for nearest-neighbor Tanimoto similarities.

    Args:
        overlap_df (pd.DataFrame): DataFrame with 'tanimoto_similarity' column

    Returns:
        pd.DataFrame: Descriptive statistics including percentiles
    """
    return (
        overlap_df["tanimoto_similarity"]
        .describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .to_frame()
    )


def novel_compounds(overlap_df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """Return query compounds that are novel (low similarity to reference).

    Args:
        overlap_df (pd.DataFrame): DataFrame with 'tanimoto_similarity' column
        threshold (float): Maximum Tanimoto similarity to consider "novel" (default: 0.4)

    Returns:
        pd.DataFrame: Query compounds with similarity below threshold
    """
    novel = overlap_df[overlap_df["tanimoto_similarity"] < threshold].copy()
    return novel.sort_values("tanimoto_similarity", ascending=True).reset_index(drop=True)


def similar_compounds(overlap_df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Return query compounds that are similar to reference (high overlap).

    Args:
        overlap_df (pd.DataFrame): DataFrame with 'tanimoto_similarity' column
        threshold (float): Minimum Tanimoto similarity to consider "similar" (default: 0.7)

    Returns:
        pd.DataFrame: Query compounds with similarity above threshold
    """
    similar = overlap_df[overlap_df["tanimoto_similarity"] >= threshold].copy()
    return similar.sort_values("tanimoto_similarity", ascending=False).reset_index(drop=True)


def overlap_fraction(overlap_df: pd.DataFrame, threshold: float = 0.7) -> float:
    """Return fraction of query compounds that overlap with reference above similarity threshold.

    Args:
        overlap_df (pd.DataFrame): DataFrame with 'tanimoto_similarity' column
        threshold (float): Minimum Tanimoto similarity to consider "overlapping"

    Returns:
        float: Fraction of query compounds with nearest neighbor similarity >= threshold
    """
    n_overlapping = (overlap_df["tanimoto_similarity"] >= threshold).sum()
    return n_overlapping / len(overlap_df)


# =============================================================================
# Statistical Summary Methods
# =============================================================================


def covariate_shift_summary(
    ref_nn_similarities: np.ndarray,
    cross_nn_similarities: np.ndarray,
) -> dict:
    """Compute all covariate shift (chemical space distribution) metrics.

    Compares the within-reference nearest-neighbor similarity distribution against
    the cross-dataset nearest-neighbor similarity distribution using KS test,
    Jensen-Shannon Divergence, and Population Stability Index.

    Args:
        ref_nn_similarities (np.ndarray): Within-reference NN similarity values
        cross_nn_similarities (np.ndarray): Cross-dataset NN similarity values

    Returns:
        dict: Dictionary with divergence metrics, severity labels, and distribution info
    """
    from workbench.utils.distribution_stats import (
        ks_test,
        jensen_shannon_divergence,
        population_stability_index,
    )

    if len(ref_nn_similarities) < 20 or len(cross_nn_similarities) < 20:
        log.warning("Small sample size (< 20 compounds) may produce unreliable divergence metrics")

    ks = ks_test(ref_nn_similarities, cross_nn_similarities)
    jsd = jensen_shannon_divergence(ref_nn_similarities, cross_nn_similarities)
    psi = population_stability_index(ref_nn_similarities, cross_nn_similarities)

    if psi < 0.1:
        psi_severity = "none"
    elif psi < 0.25:
        psi_severity = "moderate"
    else:
        psi_severity = "significant"

    return {
        "ks_statistic": ks["statistic"],
        "ks_p_value": ks["p_value"],
        "jensen_shannon_divergence": jsd,
        "population_stability_index": psi,
        "shift_detected": ks["shift_detected"],
        "psi_severity": psi_severity,
        "ref_distribution_size": len(ref_nn_similarities),
        "query_distribution_size": len(cross_nn_similarities),
        "ref_mean_similarity": float(ref_nn_similarities.mean()),
        "query_mean_similarity": float(cross_nn_similarities.mean()),
    }


def concept_shift_summary(
    alignment_df: pd.DataFrame,
    n_query_total: int,
) -> dict:
    """Compute concept shift metrics: are target values aligned where datasets overlap?

    For each query compound with sufficient structural similarity to the reference,
    compares its target value to the median target of its K nearest neighbors in the
    reference. Statistical tests assess whether residuals are centered at zero (aligned)
    or systematically shifted (concept shift / assay offset).

    Args:
        alignment_df (pd.DataFrame): Per-compound alignment with 'target_residual' column
        n_query_total (int): Total number of query compounds (for computing excluded count)

    Returns:
        dict: Dictionary with residual statistics, test results, and shift detection
    """
    from scipy.stats import ttest_1samp, wilcoxon

    residuals = alignment_df["target_residual"].values
    n_comparable = len(residuals)
    n_excluded = n_query_total - n_comparable

    if n_comparable < 5:
        log.warning(f"Only {n_comparable} comparable compounds — concept shift metrics unreliable")
        return {
            "mean_residual": float(np.mean(residuals)) if n_comparable > 0 else None,
            "median_residual": float(np.median(residuals)) if n_comparable > 0 else None,
            "std_residual": float(np.std(residuals)) if n_comparable > 0 else None,
            "mae": float(np.mean(np.abs(residuals))) if n_comparable > 0 else None,
            "rmse": float(np.sqrt(np.mean(residuals**2))) if n_comparable > 0 else None,
            "t_statistic": None,
            "t_p_value": None,
            "wilcoxon_statistic": None,
            "wilcoxon_p_value": None,
            "concept_shift_detected": None,
            "n_comparable_compounds": n_comparable,
            "n_excluded_compounds": n_excluded,
        }

    # One-sample t-test: H₀: mean residual = 0 (no systematic offset)
    t_stat, t_p = ttest_1samp(residuals, 0.0)

    # Wilcoxon signed-rank test: non-parametric H₀: median residual = 0
    nonzero_residuals = residuals[residuals != 0]
    if len(nonzero_residuals) >= 5:
        w_stat, w_p = wilcoxon(nonzero_residuals)
    else:
        w_stat, w_p = None, None

    # Concept shift detected if BOTH tests reject H₀
    if w_p is not None:
        concept_shift_detected = t_p < 0.05 and w_p < 0.05
    else:
        concept_shift_detected = t_p < 0.05

    return {
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "std_residual": float(np.std(residuals)),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "wilcoxon_statistic": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p_value": float(w_p) if w_p is not None else None,
        "concept_shift_detected": concept_shift_detected,
        "n_comparable_compounds": n_comparable,
        "n_excluded_compounds": n_excluded,
    }


# =============================================================================
# Visualization Methods
# =============================================================================


def plot_target_alignment(
    alignment_df: pd.DataFrame,
    target_column: str,
    n_query_total: int,
    figsize: tuple[int, int] = (14, 5),
) -> None:
    """Plot target alignment diagnostics: residual histogram and predicted-vs-actual scatter.

    Args:
        alignment_df (pd.DataFrame): Per-compound alignment DataFrame
        target_column (str): Name of the target column
        n_query_total (int): Total number of query compounds
        figsize (tuple[int, int]): Figure size (width, height)
    """
    import matplotlib.pyplot as plt

    if len(alignment_df) == 0:
        log.warning("No comparable compounds for target alignment plot")
        return

    residuals = alignment_df["target_residual"].values
    query_targets = alignment_df["query_target"].values
    neighbor_targets = alignment_df["neighbor_median_target"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Residual distribution
    ax1.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero (perfect alignment)")
    ax1.axvline(
        x=np.mean(residuals),
        color="darkorange",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {np.mean(residuals):.3f}",
    )
    ax1.set_xlabel("Target Residual (query − neighbor median)")
    ax1.set_ylabel("Count")
    ax1.set_title("Concept Shift: Target Residual Distribution")
    ax1.legend()

    # Annotate with summary stats
    summary = concept_shift_summary(alignment_df, n_query_total)
    textstr = (
        f"Mean: {summary['mean_residual']:.3f}\n"
        f"Median: {summary['median_residual']:.3f}\n"
        f"MAE: {summary['mae']:.3f}\n"
        f"t-test p: {summary['t_p_value']:.2e}\n"
        f"Shift: {'YES' if summary['concept_shift_detected'] else 'No'}"
    )
    ax1.text(
        0.98,
        0.98,
        textstr,
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Right: Query target vs neighbor median target (should be on diagonal if aligned)
    ax2.scatter(neighbor_targets, query_targets, alpha=0.5, s=20, color="steelblue")
    min_val = min(neighbor_targets.min(), query_targets.min())
    max_val = max(neighbor_targets.max(), query_targets.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect alignment")
    ax2.set_xlabel(f"Reference Neighbor Median ({target_column})")
    ax2.set_ylabel(f"Query ({target_column})")
    ax2.set_title("Target Value: Query vs Reference Neighbors")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_concept_shift_map(
    alignment_df: pd.DataFrame,
    df_reference: pd.DataFrame,
    df_query: pd.DataFrame,
    id_column_reference: str,
    id_column_query: str,
    df_all: pd.DataFrame,
    n_query_total: int,
    id_column: str | None = None,
    x_col: str = "x",
    y_col: str = "y",
    residual_clip: float = 15.0,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Plot concept shift on a UMAP projection of chemical space.

    Colors query compounds by their target residual relative to nearest reference
    neighbors. Shows WHERE in chemical space the datasets agree vs disagree.

    Args:
        alignment_df (pd.DataFrame): Per-compound alignment DataFrame
        df_reference (pd.DataFrame): Reference dataset
        df_query (pd.DataFrame): Query dataset
        id_column_reference (str): ID column in reference dataset
        id_column_query (str): ID column in query dataset
        df_all (pd.DataFrame): Combined DataFrame with UMAP coordinates
        n_query_total (int): Total number of query compounds
        id_column (str | None): ID column in df_all (defaults to id_column_query)
        x_col (str): Column name for UMAP x-coordinate (default: "x")
        y_col (str): Column name for UMAP y-coordinate (default: "y")
        residual_clip (float): Clip residual color scale to [-clip, +clip]
        figsize (tuple[int, int]): Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if id_column is None:
        id_column = id_column_query

    # Build lookup sets for reference and query IDs
    ref_ids = set(df_reference[id_column_reference].values)
    query_ids = set(df_query[id_column_query].values)

    # Map query IDs to their target residuals
    residual_map = dict(zip(alignment_df["id"], alignment_df["target_residual"]))

    # Split df_all into reference, query-comparable, query-excluded
    mask_ref = df_all[id_column].isin(ref_ids)
    mask_query = df_all[id_column].isin(query_ids)
    mask_comparable = df_all[id_column].isin(residual_map.keys())

    df_ref = df_all[mask_ref]
    df_query_comparable = df_all[mask_query & mask_comparable]
    df_query_excluded = df_all[mask_query & ~mask_comparable]

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Reference compounds — gray background
    ax.scatter(
        df_ref[x_col],
        df_ref[y_col],
        s=15,
        c="lightgray",
        alpha=0.5,
        zorder=1,
        label=f"Reference ({len(df_ref)})",
    )

    # 2. Excluded query compounds — no comparable reference neighbor
    if len(df_query_excluded) > 0:
        ax.scatter(
            df_query_excluded[x_col],
            df_query_excluded[y_col],
            s=25,
            c="dimgray",
            alpha=0.5,
            marker="x",
            zorder=2,
            label=f"Query — no overlap ({len(df_query_excluded)})",
        )

    # 3. Comparable query compounds — colored by target residual
    if len(df_query_comparable) > 0:
        residuals = df_query_comparable[id_column].map(residual_map).values

        # Clip residuals for color scale so outliers don't wash out the signal
        clipped = np.clip(residuals, -residual_clip, residual_clip)
        norm = TwoSlopeNorm(vmin=-residual_clip, vcenter=0, vmax=residual_clip)

        scatter = ax.scatter(
            df_query_comparable[x_col],
            df_query_comparable[y_col],
            s=40,
            c=clipped,
            cmap="RdBu_r",
            norm=norm,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
            label=f"Query — comparable ({len(df_query_comparable)})",
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Target Residual (query − reference neighbor median)")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Concept Shift Map: Target Alignment in Chemical Space")
    ax.legend(loc="upper left", framealpha=0.8)

    # Summary annotation
    summary = concept_shift_summary(alignment_df, n_query_total)
    textstr = (
        f"Median residual: {summary['median_residual']:.2f}\n"
        f"MAE: {summary['mae']:.2f}\n"
        f"Comparable: {summary['n_comparable_compounds']}\n"
        f"Shift detected: {'YES' if summary['concept_shift_detected'] else 'No'}"
    )
    ax.text(
        0.02,
        0.02,
        textstr,
        transform=ax.transAxes,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()


def plot_coverage_curve(
    cross_nn_similarities: np.ndarray,
    min_similarity: float = 0.3,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot a coverage curve showing how well the reference dataset covers the query.

    The coverage curve is a cumulative distribution: for each Tanimoto similarity
    threshold τ on the x-axis, the y-axis shows what fraction of query compounds
    have a nearest reference neighbor with similarity ≥ τ.

    Args:
        cross_nn_similarities (np.ndarray): Cross-dataset NN similarity values
        min_similarity (float): Minimum similarity threshold line (default: 0.3)
        figsize (tuple[int, int]): Figure size (width, height)
    """
    import matplotlib.pyplot as plt

    sims = np.sort(cross_nn_similarities)[::-1]  # Descending
    n = len(sims)
    fractions = np.arange(1, n + 1) / n  # Cumulative fraction
    thresholds = sims  # Each similarity value is a threshold

    fig, ax = plt.subplots(figsize=figsize)

    # Main coverage curve
    ax.plot(thresholds, fractions, color="steelblue", linewidth=2.5, label="Coverage curve")
    ax.fill_between(thresholds, fractions, alpha=0.15, color="steelblue")

    # Shade the "comparable zone" (above min_similarity)
    ax.axvline(
        x=min_similarity,
        color="darkorange",
        linestyle="--",
        linewidth=2,
        label=f"min_similarity = {min_similarity}",
    )
    ax.axvspan(min_similarity, 1.0, alpha=0.08, color="darkorange")

    # Mark key coverage fractions
    for tau, label_fmt in [(0.3, "τ≥0.3"), (0.5, "τ≥0.5"), (0.7, "τ≥0.7")]:
        frac = (cross_nn_similarities >= tau).sum() / n
        if frac > 0.01:  # Only annotate if non-trivial
            ax.plot(tau, frac, "o", color="steelblue", markersize=8, zorder=5)
            ax.annotate(
                f"{label_fmt}: {frac:.0%}",
                xy=(tau, frac),
                xytext=(tau + 0.03, frac + 0.05),
                fontsize=9,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            )

    # Comparable fraction annotation
    comparable_frac = (cross_nn_similarities >= min_similarity).sum() / n
    ax.text(
        0.98,
        0.98,
        f"Comparable: {comparable_frac:.0%} of query\n"
        f"Novel: {1 - comparable_frac:.0%} of query\n"
        f"Query compounds: {n}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=10,
    )

    ax.set_xlabel("Tanimoto Similarity Threshold (τ)", fontsize=12)
    ax.set_ylabel("Fraction of Query Covered (NN similarity ≥ τ)", fontsize=12)
    ax.set_title("Coverage Curve: Reference Dataset Coverage of Query Chemical Space", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_similarity_residual_funnel(
    alignment_df: pd.DataFrame,
    n_query_total: int,
    min_similarity: float = 0.3,
    figsize: tuple[int, int] = (12, 7),
) -> None:
    """Plot the similarity-residual funnel: target residuals vs Tanimoto similarity.

    Args:
        alignment_df (pd.DataFrame): Per-compound alignment DataFrame
        n_query_total (int): Total number of query compounds
        min_similarity (float): Minimum similarity threshold line (default: 0.3)
        figsize (tuple[int, int]): Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if len(alignment_df) == 0:
        log.warning("No comparable compounds for similarity-residual funnel plot")
        return

    sims = alignment_df["tanimoto_similarity"].values
    residuals = alignment_df["target_residual"].values

    fig, ax = plt.subplots(figsize=figsize)

    # Color points by residual magnitude
    abs_max = max(abs(residuals.min()), abs(residuals.max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    scatter = ax.scatter(
        sims,
        residuals,
        c=residuals,
        cmap="RdBu_r",
        norm=norm,
        alpha=0.5,
        s=25,
        edgecolors="none",
        zorder=2,
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Target Residual (query − reference neighbor median)")

    # Zero line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.7, zorder=1)

    # Running median + IQR band
    bin_edges = np.linspace(sims.min(), sims.max(), 15)
    bin_centers = []
    medians = []
    q25s = []
    q75s = []

    for i in range(len(bin_edges) - 1):
        mask = (sims >= bin_edges[i]) & (sims < bin_edges[i + 1])
        if mask.sum() >= 5:
            bin_residuals = residuals[mask]
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            medians.append(np.median(bin_residuals))
            q25s.append(np.percentile(bin_residuals, 25))
            q75s.append(np.percentile(bin_residuals, 75))

    if len(bin_centers) >= 3:
        bin_centers = np.array(bin_centers)
        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)

        ax.plot(bin_centers, medians, color="darkorange", linewidth=2.5, zorder=4, label="Running median")
        ax.fill_between(
            bin_centers, q25s, q75s, alpha=0.2, color="darkorange", zorder=3, label="IQR (25th–75th percentile)"
        )

    # Min similarity threshold line
    ax.axvline(
        x=min_similarity, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label=f"min_similarity = {min_similarity}"
    )

    # Summary annotation
    summary = concept_shift_summary(alignment_df, n_query_total)
    funnel_diagnosis = _diagnose_funnel(sims, residuals)
    textstr = (
        f"Median residual: {summary['median_residual']:.3f}\n"
        f"MAE: {summary['mae']:.3f}\n"
        f"Shift detected: {'YES' if summary['concept_shift_detected'] else 'No'}\n"
        f"Pattern: {funnel_diagnosis}"
    )
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=10,
    )

    ax.set_xlabel("Tanimoto Similarity (query → reference NN)", fontsize=12)
    ax.set_ylabel("Target Residual (query − reference neighbor median)", fontsize=12)
    ax.set_title("Similarity–Residual Funnel: Target Agreement vs Chemical Similarity", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_covariate_shift(
    ref_nn_similarities: np.ndarray,
    cross_nn_similarities: np.ndarray,
    bins: int = 50,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot overlaid histograms of within-reference and cross-dataset NN similarity distributions.

    Args:
        ref_nn_similarities (np.ndarray): Within-reference NN similarity values
        cross_nn_similarities (np.ndarray): Cross-dataset NN similarity values
        bins (int): Number of histogram bins
        figsize (tuple[int, int]): Figure size (width, height)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        ref_nn_similarities,
        bins=bins,
        alpha=0.5,
        label="Reference (within-dataset)",
        edgecolor="black",
        color="steelblue",
    )
    ax.hist(
        cross_nn_similarities,
        bins=bins,
        alpha=0.5,
        label="Query (cross-dataset)",
        edgecolor="black",
        color="darkorange",
    )

    ax.set_xlabel("Nearest Neighbor Tanimoto Similarity")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Covariate Shift: {len(ref_nn_similarities)} ref vs " f"{len(cross_nn_similarities)} query compounds"
    )
    ax.legend()

    summary = covariate_shift_summary(ref_nn_similarities, cross_nn_similarities)
    textstr = (
        f"KS stat: {summary['ks_statistic']:.3f} (p={summary['ks_p_value']:.2e})\n"
        f"JSD: {summary['jensen_shannon_divergence']:.4f}\n"
        f"PSI: {summary['population_stability_index']:.4f} ({summary['psi_severity']})"
    )
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# Helper Functions
# =============================================================================


def _diagnose_funnel(sims: np.ndarray, residuals: np.ndarray) -> str:
    """Diagnose the funnel pattern from similarity-residual data.

    Args:
        sims (np.ndarray): Tanimoto similarities
        residuals (np.ndarray): Target residuals

    Returns:
        str: Human-readable diagnosis of the funnel pattern
    """
    # Split into low-similarity and high-similarity halves
    median_sim = np.median(sims)
    low_mask = sims < median_sim
    high_mask = sims >= median_sim

    low_iqr = (
        np.percentile(residuals[low_mask], 75) - np.percentile(residuals[low_mask], 25)
        if low_mask.sum() > 5
        else float("inf")
    )
    high_iqr = (
        np.percentile(residuals[high_mask], 75) - np.percentile(residuals[high_mask], 25)
        if high_mask.sum() > 5
        else float("inf")
    )

    narrows = high_iqr < low_iqr * 0.7  # IQR narrows by at least 30%
    offset = abs(np.median(residuals)) > 0.5 * np.std(residuals)

    if narrows and not offset:
        return "Aligned funnel"
    elif narrows and offset:
        return "Offset funnel (systematic bias)"
    elif not narrows and not offset:
        return "Wide funnel (noisy data)"
    else:
        return "Wide + offset (incompatible)"
