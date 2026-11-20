"""Test the model performance report against a live Endpoint"""

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from workbench.api import Endpoint  # noqa: E402
from workbench.reports.model_performance import model_performance_report  # noqa: E402
from workbench.utils.endpoint_utils import get_evaluation_data  # noqa: E402


def test_model_performance_report(tmp_path):
    """Inference through the Endpoint, scored, broken out by a column, written as a PDF"""
    df = get_evaluation_data(Endpoint("abalone-regression")).head(300)
    path = tmp_path / "abalone_performance.pdf"
    report = model_performance_report("abalone-regression", df, path, by="sex", key="auto_id", publish=False)

    assert path.read_bytes().startswith(b"%PDF")
    breakout = report.frames["performance_breakout"]
    assert breakout.loc[0, "group"] == "all" and breakout.loc[0, "support"] == 300
    assert set(breakout["group"][1:]) <= {"M", "F", "I", "other"}
    rows = report.frames["performance_rows"]
    assert list(rows.columns) == ["auto_id", "sex", "class_number_of_rings", "prediction", "error"]
    assert rows["error"].abs().is_monotonic_decreasing


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--no-cov"])
