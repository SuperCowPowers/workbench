from sageworks.api.endpoint import Endpoint
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

# Grab an existing Endpoint
endpoint = Endpoint("abalone-regression-end")

# SageWorks tracks both Model performance and Endpoint Metrics
model_metrics = endpoint.details()["model_metrics"]
endpoint_metrics = endpoint.endpoint_metrics()
print(model_metrics)
print(endpoint_metrics.iloc[:, 1:6])
