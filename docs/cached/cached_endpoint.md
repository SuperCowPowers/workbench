# CachedEndpoint

!!! tip inline end "Model Examples"
    Examples of using the Model Class are in the [Examples](#examples) section at the bottom of this page. AWS Model setup and deployment are quite complicated to do manually but the Workbench Model Class makes it a breeze!

::: workbench.cached.cached_endpoint


## Examples
All of the Workbench Examples are in the Workbench Repository under the `examples/` directory. For a full code listing of any example please visit our [Workbench Examples](https://github.com/SuperCowPowers/workbench/blob/main/examples)

**Get Endpoint Details**

```python
from workbench.cached.cached_endpoint import CachedEndpoint

# Grab an Endpoint
end = CachedEndpoint("abalone-regression")

# Get the Details
 end.details()

{'uuid': 'abalone-regression-end',
 'health_tags': [],
 'status': 'InService',
 'instance': 'Serverless (2GB/5)',
 'instance_count': '-',
 'variant': 'AllTraffic',
 'model_name': 'abalone-regression',
 'model_type': 'regressor',
 'model_metrics':        RMSE     R2    MAPE  MedAE  NumRows
 1.64  2.246  0.502  16.393  1.209      834,
 'confusion_matrix': None,
 'predictions':      class_number_of_rings  prediction    id
 0                       16   10.516158     7
 1                        9    9.031365     8
 2                       10    9.264600    17
 3                        7    8.578638    18
 4                       12   10.492446    27
 ..                     ...         ...   ...
 829                     11   11.915862  4148
 830                      8    8.210898  4157
 831                      8    7.693689  4158
 832                      9    7.542521  4167
 833                      8    9.060015  4168
```
