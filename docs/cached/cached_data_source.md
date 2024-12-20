# CachedDataSource

!!! tip inline end "Model Examples"
    Examples of using the Model Class are in the [Examples](#examples) section at the bottom of this page. AWS Model setup and deployment are quite complicated to do manually but the Workbench Model Class makes it a breeze!

::: workbench.cached.cached_data_source


## Examples
All of the Workbench Examples are in the Workbench Repository under the `examples/` directory. For a full code listing of any example please visit our [Workbench Examples](https://github.com/SuperCowPowers/workbench/blob/main/examples)

**Pull DataSource Details**

```python
from workbench.cached.cached_data_source import CachedDataSource

# Grab a DataSource
ds = CachedDataSource("abalone_data")

# Show the details
ds.details()

> ds.details()

{'uuid': 'abalone_data',
 'health_tags': [],
 'aws_arn': 'arn:aws:glue:x:table/workbench/abalone_data',
 'size': 0.070272,
 'created': '2024-11-09T20:42:34.000Z',
 'modified': '2024-11-10T19:57:52.000Z',
 'input': 's3://workbench-public-data/common/aBaLone.CSV',
 'workbench_health_tags': '',
 'workbench_correlations': {'length': {'diameter': 0.9868115846024996,

```
