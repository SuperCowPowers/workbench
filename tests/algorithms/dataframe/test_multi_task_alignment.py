"""Unit tests for MultiTaskAlignment scoring.

Each test pins one property of the score: verdicts depend on the strength of a
relationship and not its direction, the residual measures disagreement rather than
direction, coverage resolves against the primary set at any sparsity, and the verdict
thresholds apply to rank correlation.
"""

import numpy as np
import pandas as pd

from workbench.algorithms.dataframe.multi_task_alignment import MultiTaskAlignment

# Chains and rings of varying length and substitution, for a graded Tanimoto structure
# rather than a set of near-duplicates.
TAILS = ["O", "N", "C(=O)O", "c1ccccc1", "c1ccncc1", "S", "C#N", "C(F)(F)F", "OC", "N(C)C"]
SMILES = [f"{'C' * n}{tail}" for tail in TAILS for n in range(1, 21)]

PRIMARY_SEED = 42
AUX_SEED = 7  # distinct from PRIMARY_SEED, or the "noise" is a linear function of the primary


def alignment_frame(aux_fn, n_primary: int = 40) -> pd.DataFrame:
    """Wide multi-task frame: every row has the aux, the first `n_primary` have the primary."""
    primary = np.random.default_rng(PRIMARY_SEED).normal(5.0, 1.0, len(SMILES))
    return pd.DataFrame(
        {
            "id": [f"c{i}" for i in range(len(SMILES))],
            "smiles": SMILES,
            "primary": [v if i < n_primary else np.nan for i, v in enumerate(primary)],
            "aux": aux_fn(primary),
        }
    )


def summary_for(aux_fn, **kwargs) -> pd.Series:
    df = alignment_frame(aux_fn, **kwargs)
    mta = MultiTaskAlignment(df, primary="primary", auxiliaries=["aux"], id_column="id", min_n_shared=10)
    return mta.summary().iloc[0]


def noisy(scale: float):
    """An aux that tracks the primary with noise, in a given direction."""
    return lambda p: scale * p + np.random.default_rng(AUX_SEED).normal(0, 0.9, len(p))


def test_verdict_depends_on_strength_not_direction():
    """A shared encoder exploits a relationship in either direction, so |r| sets the verdict.

    Efficacy and fold-change arms read lower as potency rises; they carry the same signal as
    a potency readout and score the same.
    """
    positive = summary_for(noisy(1.0))
    negative = summary_for(noisy(-1.0))

    assert negative["spearman_r"] < 0 < positive["spearman_r"]
    assert negative["overlap"] == positive["overlap"] == "Beneficial"


def test_residual_measures_disagreement_not_direction():
    """The residual scores how far an aux sits from its neighborhood, not which way it points."""
    positive = summary_for(noisy(1.0))
    negative = summary_for(noisy(-1.0))

    assert negative["residual_abs_mean"] == approx(positive["residual_abs_mean"], 0.05)


def test_coverage_resolves_against_a_sparse_primary_set():
    """Every row reports its true nearest primary neighbor, at any primary sparsity.

    Five primary rows against 200 compounds: coverage is a similarity to a real neighbor,
    never a placeholder for one that was not looked up.
    """
    df = alignment_frame(lambda p: p, n_primary=5)
    mta = MultiTaskAlignment(df, primary="primary", auxiliaries=["aux"], id_column="id", min_n_shared=1)
    coverage = mta.results()["tanimoto_to_primary"]

    assert len(coverage) == len(SMILES)
    assert coverage.notna().all()
    assert (coverage > 0).mean() > 0.95


def test_thresholds_apply_to_rank_correlation():
    """A monotone but saturating aux is usable, and the verdict reflects that.

    Percent-of-control and fold-change readouts saturate at both ends. Noise applied before
    the transform keeps the relationship monotone while flattening the linear correlation,
    so the two statistics straddle the 0.4 cutoff and the verdict follows the rank one.
    """

    def saturating(p):
        return 30.0 ** (p + np.random.default_rng(AUX_SEED).normal(0, 1.0, len(p)))

    row = summary_for(saturating)

    assert abs(row["pearson_r"]) < 0.4 <= abs(row["spearman_r"])
    assert row["overlap"] == "Beneficial"


def approx(value: float, tol: float):
    """Absolute-tolerance comparison helper."""

    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= tol

        def __repr__(self):
            return f"{value} +/- {tol}"

    return _Approx()
