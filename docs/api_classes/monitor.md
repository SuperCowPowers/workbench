# Monitor

!!! tip inline end "Monitor Examples"
    Examples of using the Monitor class are listed at the bottom of this page [Examples](#examples).
    
::: workbench.api.monitor


## Examples

**Initial Setup of the Endpoint Monitor**

```py title="monitor_setup.py"
from workbench.api.monitor import Monitor

# Create an Endpoint Monitor Class and perform initial Setup
endpoint_name = "abalone-regression-end-rt"
mon = Monitor(endpoint_name)

# Add data capture to the endpoint
mon.add_data_capture(capture_percentage=100)

# Create a baseline for monitoring
mon.create_baseline()

# Set up the monitoring schedule
mon.create_monitoring_schedule(schedule="hourly")
```

**Pulling Information from an Existing Monitor**

```py title="monitor_usage.py"
from workbench.api.monitor import Monitor
from workbench.api.endpoint import Endpoint

# Construct a Monitor Class in one of Two Ways
mon = Endpoint("abalone-regression-end-rt").get_monitor()
mon = Monitor("abalone-regression-end-rt")

# Check the summary and details of the monitoring class
mon.summary()
mon.details()

# Check the baseline outputs (baseline, constraints, statistics)
base_df = mon.get_baseline()
base_df.head()

constraints_df = mon.get_constraints()
constraints_df.head()

statistics_df = mon.get_statistics()
statistics_df.head()

# Get the latest data capture (inputs and outputs)
input_df, output_df = mon.get_latest_data_capture()
input_df.head()
output_df.head()
```


## Workbench UI
Running these few lines of code creates and deploys an AWS Endpoint Monitor. The Monitor status and outputs can be viewed in the Sagemaker Console interfaces or in the Workbench Dashboard UI. Workbench will use the monitor to track various metrics including Data Quality, Model Bias, etc...

<figure>
<img alt="workbench_endpoints" src="https://github.com/SuperCowPowers/workbench/assets/4806709/b5eab741-2c23-4c5e-9495-15fd3ea8155c">
<figcaption>Workbench Dashboard: Endpoints</figcaption>
</figure>


!!! note "Not Finding a particular method?"
    The Workbench API Classes use the 'Core' Classes Internally, so for an extensive listing of all the methods available please take a deep dive into: [Workbench Core Classes](../core_classes/overview.md)