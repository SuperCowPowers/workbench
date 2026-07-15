"""Reports: Published analysis reports (DataFrames) stored in the Workbench DataFrame Store"""

# Workbench Imports
from workbench.api.df_store import DFStore


class Reports(DFStore):
    """Reports: Published analysis reports stored under the /reports subtree of the DataFrame Store.

    A thin wrapper over DFStore that scopes every operation (list/get/upsert/delete)
    to the /reports subtree. Writers publish reports (e.g. the promotion arbiter
    publishing contest results); readers (dashboards, scripts) list and get them.
    Reads are uncached: every get() hits S3.

    Common Usage:
        ```python
        reports = Reports()

        # List all published reports
        reports.list()

        # Publish a report
        reports.upsert("/contests/my-endpoint", ranked_df)

        # Retrieve a report
        df = reports.get("/contests/my-endpoint")

        # Delete a report
        reports.delete("/contests/my-endpoint")
        ```
    """

    def __init__(self):
        """Reports Init: a DFStore scoped to the /reports subtree"""
        super().__init__(path_prefix="/reports")


if __name__ == "__main__":
    """Exercise the Reports Class"""
    import pandas as pd

    reports = Reports()

    # Publish a report
    my_report = pd.DataFrame({"model": ["model-1", "model-2"], "rmse": [0.68, 0.71]})
    reports.upsert("/testing/example_report", my_report)

    # List the published reports
    print("Published Reports:")
    print(reports.list())

    # Retrieve the report
    report = reports.get("/testing/example_report")
    print(report)

    # Reports only sees the /reports subtree (not the rest of the DataFrame Store)
    print(f"Full DFStore locations: {len(DFStore().list())}")
    print(f"Reports locations: {len(reports.list())}")

    # Clean up
    reports.delete("/testing/example_report")
    print(f"After delete: {reports.list()}")
