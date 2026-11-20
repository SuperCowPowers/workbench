from workbench.api import Endpoint
from workbench.reports.model_performance import model_performance_report
from workbench.utils.endpoint_utils import get_evaluation_data

# Rows the model was not trained on, with the measured target
endpoint_name = "abalone-regression"
df = get_evaluation_data(Endpoint(endpoint_name))

# Run inference, score it, and write the PDF; performance is also broken out per sex
report = model_performance_report(endpoint_name, df, "abalone_performance.pdf", by="sex", key="auto_id")

# The numbers behind the pages
print(report.frames["performance_breakout"])
print(report.frames["performance_rows"].head())
