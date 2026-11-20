"""Model performance report for an Endpoint: run inference on a DataFrame, score it, write the PDF"""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from workbench.api import Endpoint, Model
from workbench.reports import sections
from workbench.reports.report import Report
from workbench.utils.metrics_utils import resolve_primary_target


def model_performance_report(
    endpoint_name: str,
    df: pd.DataFrame,
    path: Union[str, Path],
    target: Optional[str] = None,
    by: Optional[str] = None,
    key: Union[str, List[str], None] = None,
    log: bool = False,
    unit: Optional[str] = None,
    overwrite: bool = False,
    publish: bool = True,
) -> Report:
    """Run a DataFrame through an Endpoint and write a report on how well it predicts.

    Args:
        endpoint_name (str): The Endpoint to run inference on.
        df (pd.DataFrame): Rows to predict, with the model's features and the measured target.
        path (Union[str, Path]): Where to write the PDF.
        target (Optional[str]): Measured value column. Defaults to the model's target.
        by (Optional[str]): Column to break performance out on, e.g. "project".
        key (Union[str, List[str], None]): Column(s) identifying a row. Defaults to the index.
        log (bool): The target is a positive quantity: log axes and fold-error metrics.
        unit (Optional[str]): Target unit for axis labels.
        overwrite (bool): Replace an existing PDF. Defaults to False.
        publish (bool): Publish the rows behind the report to the DataFrame Store. Defaults to True.

    Returns:
        Report: The saved report; `report.frames` holds the scored rows and the metrics table.
    """
    endpoint = Endpoint(endpoint_name)
    target = target or resolve_primary_target(Model(endpoint.get_input()).target())
    predictions = endpoint.inference(df)

    report = Report(f"{endpoint_name} performance")
    report.add(sections.performance(predictions, target, by=by, key=key, log=log, unit=unit))
    report.save(path, overwrite=overwrite, publish=publish)
    return report
