# CachedModel

!!! tip inline end "Model Examples"
    Examples of using the Model Class are in the [Examples](#examples) section at the bottom of this page. AWS Model setup and deployment are quite complicated to do manually but the SageWorks Model Class makes it a breeze!

::: sageworks.cached.cached_model


## Examples
All of the SageWorks Examples are in the Sageworks Repository under the `examples/` directory. For a full code listing of any example please visit our [SageWorks Examples](https://github.com/SuperCowPowers/sageworks/blob/main/examples)

**Pull Inference Run**

```python
from sageworks.cached.cached_model import CachedModel

# Grab a Model
model = CachedModel("abalone-regression")

# List the inference runs
model.list_inference_runs()
['auto_inference', 'model_training']

# Grab specific inference results
model.get_inference_predictions("auto_inference")
     class_number_of_rings  prediction    id
0                       16   10.516158     7
1                        9    9.031365     8
..                     ...         ...   ...
831                      8    7.693689  4158
832                      9    7.542521  4167

```
