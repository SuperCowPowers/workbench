"""Unit tests for target_health — regression-target column pathology checks."""

import numpy as np
import pandas as pd

from workbench.utils.pandas_utils import target_health


def severity(df: pd.DataFrame, check: str) -> str:
    """Severity reported for a single check."""
    return df.loc[df["check"] == check, "severity"].iloc[0]


def frame(values) -> pd.DataFrame:
    return pd.DataFrame({"target": values})


def test_censoring_flags_a_stack_on_the_maximum():
    """Rows piled on the exact max are the signature of a clipped assay."""
    values = list(np.linspace(0, 90, 80)) + [100.0] * 20
    result = target_health(frame(values), "target")

    assert severity(result, "censoring") == "warn"
    assert "100" in result.loc[result["check"] == "censoring", "value"].iloc[0]


def test_censoring_ignores_a_bare_distribution_tail():
    """A single row at the max is the end of a distribution, not a pileup."""
    result = target_health(frame(list(np.linspace(0, 100, 100))), "target")

    assert severity(result, "censoring") == "ok"


def test_censoring_ignores_tiny_sets():
    """On five rows, one value at the min is 20% — the row floor keeps that quiet."""
    result = target_health(frame([-0.5, 0.1, 1.5, 3.0, 2.0]), "target")

    assert severity(result, "censoring") == "ok"
    assert severity(result, "pileup") == "ok"


def test_discretization_flags_a_coarse_reporting_grid():
    """A target rounded to few distinct values caps achievable error."""
    result = target_health(frame([round(v, 1) for v in np.linspace(0, 5, 2000)]), "target")

    assert severity(result, "discretization") == "warn"


def test_discretization_ok_for_continuous_values():
    """Genuinely continuous targets don't trip the check."""
    result = target_health(frame(list(np.linspace(0, 5, 500))), "target")

    assert severity(result, "discretization") == "ok"


def test_skew_warns_and_notes_non_positive_values():
    """A log-normal target needs a transform, and zeros need an offset first."""
    values = [0.0] * 10 + [1.0] * 200 + [500.0] * 5
    result = target_health(frame(values), "target")

    assert severity(result, "skew") == "warn"
    assert "offset" in result.loc[result["check"] == "skew", "detail"].iloc[0]


def test_skew_ok_for_symmetric_targets():
    """A roughly symmetric target needs no transform."""
    result = target_health(frame(list(np.linspace(-3, 3, 200))), "target")

    assert severity(result, "skew") == "ok"


def test_missing_targets_are_reported():
    """NaN targets are counted and flagged."""
    result = target_health(frame([1.0, 2.0, np.nan, 4.0]), "target")

    assert severity(result, "missing") == "warn"
    assert result.loc[result["check"] == "missing", "value"].iloc[0].startswith("1 ")


def test_all_missing_returns_only_the_missing_check():
    """With nothing to analyze the checks stop rather than raise."""
    result = target_health(frame([np.nan, np.nan]), "target")

    assert result["check"].tolist() == ["missing"]


def test_checks_ignore_nans_when_computing_range():
    """The range reflects the observed values, not the NaN holes."""
    result = target_health(frame([1.0, np.nan, 5.0, 3.0]), "target")

    assert result.loc[result["check"] == "range", "value"].iloc[0] == "[1, 5]"
