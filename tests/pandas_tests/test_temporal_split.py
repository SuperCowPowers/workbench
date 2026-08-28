import pandas as pd
import numpy as np

from workbench.utils.pandas_utils import temporal_split


def make_df(n=100):
    """Create a simple DataFrame with an id and date column."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({"id": range(n), "date": dates})


class TestTemporalSplit:
    """Tests for the date-boundary temporal split."""

    def test_basic_split(self):
        df = make_df(100)  # 2023-01-01 to 2023-04-10
        train, test = temporal_split(df, "date", end_date="2023-03-01")
        assert (train["date"] <= pd.Timestamp("2023-03-01")).all()
        assert (test["date"] > pd.Timestamp("2023-03-01")).all()

    def test_no_row_loss(self):
        df = make_df(50)
        train, test = temporal_split(df, "date", end_date="2023-02-01")
        all_ids = set(train["id"]).union(set(test["id"]))
        assert all_ids == set(range(50))

    def test_train_dates_before_test(self):
        df = make_df(100)
        train, test = temporal_split(df, "date", end_date="2023-02-15")
        assert train["date"].max() <= test["date"].min()

    def test_end_date_after_all_data(self):
        df = make_df(100)
        train, test = temporal_split(df, "date", end_date="2025-01-01")
        assert len(train) == 100
        assert len(test) == 0

    def test_end_date_before_all_data(self):
        df = make_df(100)
        train, test = temporal_split(df, "date", end_date="2020-01-01")
        assert len(train) == 0
        assert len(test) == 100


class TestTemporalSplitEdgeCases:
    """Tests for edge cases and data quality."""

    def test_nat_handling(self):
        """A date-less row carries no temporal information, so it trains and is never held out."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "date": ["2023-01-01", "2023-02-01", "not-a-date", "2023-04-01"]})
        train, test = temporal_split(df, "date", end_date="2023-02-15")
        assert set(train["id"]) == {1, 2, 3}
        assert set(test["id"]) == {4}

    def test_all_dates_null(self):
        """An entirely date-less column holds nothing out."""
        df = pd.DataFrame({"id": [1, 2, 3], "date": [None, None, None]})
        train, test = temporal_split(df, "date", end_date="2023-02-15")
        assert len(train) == 3
        assert len(test) == 0

    def test_string_dates_parsed(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]})
        train, test = temporal_split(df, "date", end_date="2023-02-15")
        assert len(train) == 2
        assert len(test) == 2

    def test_exact_boundary_date(self):
        """Rows on the exact end_date should be in train (<=), not test (>)."""
        df = pd.DataFrame({"id": [1, 2, 3], "date": ["2023-01-01", "2023-02-01", "2023-03-01"]})
        train, test = temporal_split(df, "date", end_date="2023-02-01")
        assert set(train["id"]) == {1, 2}
        assert set(test["id"]) == {3}
