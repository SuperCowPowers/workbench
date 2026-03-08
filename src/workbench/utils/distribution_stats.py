"""Distribution comparison statistics for detecting covariate shift.

Standalone utility functions for comparing two distributions using standard
statistical tests. These are general-purpose and not specific to any particular
domain (fingerprints, molecular data, etc.).

Functions:
    ks_test: Kolmogorov-Smirnov two-sample test
    jensen_shannon_divergence: Jensen-Shannon Divergence (symmetric, bounded)
    population_stability_index: PSI for monitoring distribution drift
"""

import numpy as np


def _compute_binned_distributions(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    n_bins: int = 10,
    range_min: float = 0.0,
    range_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized binned distributions for two arrays.

    Histograms both distributions into the same bins, adds epsilon to avoid
    zero bins, and normalizes. Default of 10 bins (deciles) follows standard
    PSI practice and avoids sparse-bin artifacts with typical dataset sizes.

    Args:
        dist_a (np.ndarray): First distribution (reference/baseline)
        dist_b (np.ndarray): Second distribution (query/comparison)
        n_bins (int): Number of histogram bins
        range_min (float): Minimum bin edge (default: 0.0)
        range_max (float): Maximum bin edge (default: 1.0)

    Returns:
        tuple[np.ndarray, np.ndarray]: (p, q) normalized distributions summing to 1
    """
    bins = np.linspace(range_min, range_max, n_bins + 1)
    epsilon = 1e-10

    p_counts, _ = np.histogram(dist_a, bins=bins)
    q_counts, _ = np.histogram(dist_b, bins=bins)

    p = (p_counts + epsilon) / (p_counts + epsilon).sum()
    q = (q_counts + epsilon) / (q_counts + epsilon).sum()

    return p, q


def ks_test(dist_a: np.ndarray, dist_b: np.ndarray) -> dict:
    """Kolmogorov-Smirnov two-sample test comparing two distributions.

    The KS test measures the maximum distance between two empirical CDFs.
    A large test statistic (or small p-value) indicates the distributions
    are drawn from different underlying populations.

    Args:
        dist_a (np.ndarray): First distribution (reference)
        dist_b (np.ndarray): Second distribution (query)

    Returns:
        dict: Keys are 'statistic' (float, 0-1), 'p_value' (float),
            and 'shift_detected' (bool, True if p_value < 0.05)
    """
    from scipy.stats import ks_2samp

    stat, p_value = ks_2samp(dist_a, dist_b)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "shift_detected": p_value < 0.05,
    }


def jensen_shannon_divergence(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Jensen-Shannon Divergence between two distributions.

    JSD is a symmetric, bounded (0 to 1) divergence measure. Values near 0
    indicate similar distributions; values near 1 indicate completely different.

    Args:
        dist_a (np.ndarray): First distribution (reference)
        dist_b (np.ndarray): Second distribution (query)
        n_bins (int): Number of bins for histogram comparison

    Returns:
        float: JSD value in [0, 1]
    """
    from scipy.spatial.distance import jensenshannon

    p, q = _compute_binned_distributions(dist_a, dist_b, n_bins=n_bins)
    # scipy returns sqrt(JSD) (the JS distance), so square it for true divergence
    # base=2 ensures JSD is bounded [0, 1]
    return float(jensenshannon(p, q, base=2) ** 2)


def population_stability_index(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index comparing two distributions.

    PSI quantifies how much a distribution has shifted from a baseline.
    Commonly used in model monitoring to detect covariate shift.

    Interpretation:
        PSI < 0.1:  No significant shift
        0.1 <= PSI < 0.25:  Moderate shift, investigation recommended
        PSI >= 0.25:  Significant shift, action required

    Args:
        dist_a (np.ndarray): Baseline distribution (reference)
        dist_b (np.ndarray): Comparison distribution (query)
        n_bins (int): Number of bins for histogram comparison

    Returns:
        float: PSI value (non-negative)
    """
    p, q = _compute_binned_distributions(dist_a, dist_b, n_bins=n_bins)
    return float(np.sum((q - p) * np.log(q / p)))
