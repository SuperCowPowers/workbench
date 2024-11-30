# CachedFeatureSet

!!! tip inline end "Model Examples"
    Examples of using the Model Class are in the [Examples](#examples) section at the bottom of this page. AWS Model setup and deployment are quite complicated to do manually but the SageWorks Model Class makes it a breeze!

::: sageworks.cached.cached_feature_set


## Examples
All of the SageWorks Examples are in the Sageworks Repository under the `examples/` directory. For a full code listing of any example please visit our [SageWorks Examples](https://github.com/SuperCowPowers/sageworks/blob/main/examples)

**Pull FeatureSet Details**

```python
from sageworks.cached.cached_feature_set import CachedFeatureSet

# Grab a FeatureSet
fs = CachedFeatureSet("abalone_features")

# Show the details
fs.details()

> fs.details()

{'uuid': 'abalone_features',
 'health_tags': [],
 'aws_arn': 'arn:aws:glue:x:table/sageworks/abalone_data',
 'size': 0.070272,
 'created': '2024-11-09T20:42:34.000Z',
 'modified': '2024-11-10T19:57:52.000Z',
 'input': 's3://sageworks-public-data/common/aBaLone.CSV',
 'sageworks_health_tags': '',
 'sageworks_correlations': {'length': {'diameter': 0.9868115846024996,

```