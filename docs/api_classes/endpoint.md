# Endpoint

!!! tip inline end "Endpoint Examples"
    Examples of using the Endpoint class are listed at the bottom of this page [Examples](#examples).
    
::: sageworks.api.endpoint


## Examples

**Run Inference on an Endpoint**

```py title="endpoint_inference.py"
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model
from sageworks.api.endpoint import Endpoint

# Grab an existing Endpoint
endpoint = Endpoint("abalone-regression-end")

# SageWorks has full ML Pipeline provenance, so we can backtrack the inputs,
# get a DataFrame of data (not used for training) and run inference
model = Model(endpoint.get_input())
fs = FeatureSet(model.get_input())
athena_table = fs.view("training").table
df = fs.query(f"SELECT * FROM {athena_table} where training = 0")

# Run inference/predictions on the Endpoint
results_df = endpoint.inference(df)

# Run inference/predictions and capture the results
results_df = endpoint.inference(df, capture=True)

# Run inference/predictions using the FeatureSet evaluation data
results_df = endpoint.auto_inference(capture=True)
```

**Output**

```py
Processing...
     class_number_of_rings  prediction
0                       13   11.477922
1                       12   12.316887
2                        8    7.612847
3                        8    9.663341
4                        9    9.075263
..                     ...         ...
839                      8    8.069856
840                     15   14.915502
841                     11   10.977605
842                     10   10.173433
843                      7    7.297976
```
**Endpoint Details**

!!!tip inline end "The details() method"
    The `detail()` method on the Endpoint class provides a lot of useful information. All of the SageWorks classes have a `details()` method try it out!

```py title="endpoint_details.py"
from sageworks.api.endpoint import Endpoint
from pprint import pprint

# Get Endpoint and print out it's details
endpoint = Endpoint("abalone-regression-end")
pprint(endpoint.details())
```

**Output**

```py
{
 'input': 'abalone-regression',
 'instance': 'Serverless (2GB/5)',
 'model_metrics':   metric_name  value
			0        RMSE  2.190
			1         MAE  1.544
			2          R2  0.504,
 'model_name': 'abalone-regression',
 'model_type': 'regressor',
 'modified': datetime.datetime(2023, 12, 29, 17, 48, 35, 115000, tzinfo=datetime.timezone.utc),
     class_number_of_rings  prediction
0                        9    8.648378
1                       11    9.717787
2                       11   10.933070
3                       10    9.899738
4                        9   10.014504
..                     ...         ...
495                     10   10.261657
496                      9   10.788254
497                     13    7.779886
498                     12   14.718514
499                     13   10.637320
 'sageworks_tags': ['abalone', 'regression'],
 'status': 'InService',
 'uuid': 'abalone-regression-end',
 'variant': 'AllTraffic'}
```

**Endpoint Metrics**

```py title="endpoint_metrics.py"
from sageworks.api.endpoint import Endpoint

# Grab an existing Endpoint
endpoint = Endpoint("abalone-regression-end")

# SageWorks tracks both Model performance and Endpoint Metrics
model_metrics = endpoint.details()["model_metrics"]
endpoint_metrics = endpoint.endpoint_metrics()
print(model_metrics)
print(endpoint_metrics)
```

**Output**

```py
  metric_name  value
0        RMSE  2.190
1         MAE  1.544
2          R2  0.504

    Invocations  ModelLatency  OverheadLatency  ModelSetupTime  Invocation5XXErrors
29          0.0          0.00             0.00            0.00                  0.0
30          1.0          1.11            23.73           23.34                  0.0
31          0.0          0.00             0.00            0.00                  0.0
48          0.0          0.00             0.00            0.00                  0.0
49          5.0          0.45             9.64           23.57                  0.0
50          2.0          0.57             0.08            0.00                  0.0
51          0.0          0.00             0.00            0.00                  0.0
60          4.0          0.33             5.80           22.65                  0.0
61          1.0          1.11            23.35           23.10                  0.0
62          0.0          0.00             0.00            0.00                  0.0
...
```


## SageWorks UI
Running these few lines of code creates and deploys an AWS Endpoint. The Endpoint artifacts can be viewed in the Sagemaker Console/Notebook interfaces or in the SageWorks Dashboard UI. SageWorks will monitor the endpoint, plot invocations, latencies, and tracks error metrics.

<figure>
<img alt="sageworks_endpoints" src="https://github.com/SuperCowPowers/sageworks/assets/4806709/b5eab741-2c23-4c5e-9495-15fd3ea8155c">
<figcaption>SageWorks Dashboard: Endpoints</figcaption>
</figure>


!!! note "Not Finding a particular method?"
    The SageWorks API Classes use the 'Core' Classes Internally, so for an extensive listing of all the methods available please take a deep dive into: [SageWorks Core Classes](../core_classes/overview.md)
