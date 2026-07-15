"""Tests for the Reports class (DFStore scoped to the /reports subtree)"""

import pandas as pd

# Workbench Imports
from workbench.api import DFStore, Reports


def test_reports_roundtrip():
    """Reports upsert/get/list/delete round-trip"""
    reports = Reports()
    report = pd.DataFrame({"model": ["model-1", "model-2"], "rmse": [0.68, 0.71]})
    reports.upsert("/testing/roundtrip_report", report)

    assert "/testing/roundtrip_report" in reports.list()
    fetched = reports.get("/testing/roundtrip_report")
    assert list(fetched["model"]) == ["model-1", "model-2"]

    reports.delete("/testing/roundtrip_report")
    assert "/testing/roundtrip_report" not in reports.list()


def test_reports_scoped_to_subtree():
    """Reports only sees the /reports subtree, and its data lands there in the full DFStore"""
    reports = Reports()
    report = pd.DataFrame({"A": [1]})
    reports.upsert("/testing/scope_report", report)

    # Visible under /reports in the full DFStore, prefix-relative in Reports
    assert "/reports/testing/scope_report" in DFStore().list()
    assert "/testing/scope_report" in reports.list()

    # Reports.list() shows nothing outside the /reports subtree
    assert all(not loc.startswith("/reports") for loc in reports.list())

    reports.delete("/testing/scope_report")


if __name__ == "__main__":
    test_reports_roundtrip()
    test_reports_scoped_to_subtree()
    print("All Reports tests passed!")
