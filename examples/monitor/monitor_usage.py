from pprint import pprint
from workbench.api.monitor import Monitor
from workbench.api.endpoint import Endpoint

# Construct a Monitor Class in one of Two Ways
mon = Endpoint("abalone-regression-end-rt").monitor()
mon = Monitor("abalone-regression-end-rt")

# Check the summary and details of the monitoring class
pprint(mon.summary())
pprint(mon.details())

# Check the baseline outputs (baseline, constraints, statistics)
base_df = mon.get_baseline()
print(base_df.head())

constraints_df = mon.get_constraints()
print(constraints_df.head())

statistics_df = mon.get_statistics()
print(statistics_df.head())
