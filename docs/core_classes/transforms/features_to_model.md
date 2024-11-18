# Features To Model
!!! tip inline end "API Classes"
    For most users the [API Classes](../../api_classes/overview.md) will provide all the general functionality to create a full AWS ML Pipeline

::: sageworks.core.transforms.features_to_model.features_to_model

## Supported Models
Currently SageWorks supports XGBoost (classifier/regressor), and Scikit Learn models. Those models can be created by just specifying different parameters to the `FeaturesToModel` class. The main issue with the supported models is they are vanilla versions with default parameters, any customization should be done with [Custom Models](#custom-models)

### XGBoost
```python
from sageworks.core.transforms.features_to_model.features_to_model import FeaturesToModel

# XGBoost Regression Model
input_uuid = "abalone_features"
output_uuid = "abalone-regression"
to_model = FeaturesToModel(input_uuid, output_uuid, model_type=ModelType.REGRESSOR)
to_model.set_output_tags(["abalone", "public"])
to_model.transform(target_column="class_number_of_rings", description="Abalone Regression")

# XGBoost Classification Model
input_uuid = "wine_features"
output_uuid = "wine-classification"
to_model = FeaturesToModel(input_uuid, output_uuid, ModelType.CLASSIFIER)
to_model.set_output_tags(["wine", "public"])
to_model.transform(target_column="wine_class", description="Wine Classification")

# Quantile Regression Model (Abalone)
input_uuid = "abalone_features"
output_uuid = "abalone-quantile-reg"
to_model = FeaturesToModel(input_uuid, output_uuid, ModelType.QUANTILE_REGRESSOR)
to_model.set_output_tags(["abalone", "quantiles"])
to_model.transform(target_column="class_number_of_rings", description="Abalone Quantile Regression")
```
### Scikit-Learn
```python
from sageworks.core.transforms.features_to_model.features_to_model import FeaturesToModel

# Scikit-Learn Kmeans Clustering Model
input_uuid = "wine_features"
output_uuid = "wine-clusters"
to_model = FeaturesToModel(
    input_uuid,
    output_uuid,
    model_class="KMeans",  # Clustering algorithm
    model_import_str="from sklearn.cluster import KMeans",  # Import statement for KMeans
    model_type=ModelType.CLUSTERER,
)
to_model.set_output_tags(["wine", "clustering"])
to_model.transform(target_column=None, description="Wine Clustering", train_all_data=True)

# Scikit-Learn HDBSCAN Clustering Model
input_uuid = "wine_features"
output_uuid = "wine-clusters-hdbscan"
to_model = FeaturesToModel(
    input_uuid,
    output_uuid,
    model_class="HDBSCAN",  # Density-based clustering algorithm
    model_import_str="from sklearn.cluster import HDBSCAN",
    model_type=ModelType.CLUSTERER,
)
to_model.set_output_tags(["wine", "density-based clustering"])
to_model.transform(target_column=None, description="Wine Clustering with HDBSCAN", train_all_data=True)

# Scikit-Learn 2D Projection Model using UMAP
input_uuid = "wine_features"
output_uuid = "wine-2d-projection"
to_model = FeaturesToModel(
    input_uuid,
    output_uuid,
    model_class="UMAP",
    model_import_str="from umap import UMAP",
    model_type=ModelType.PROJECTION,
)
to_model.set_output_tags(["wine", "2d-projection"])
to_model.transform(target_column=None, description="Wine 2D Projection", train_all_data=True)
```
    
## Custom Models
For custom models we recommend the following steps:

!!! warning inline end "Experimental"
    The SageWorks Custom Models are currently in experimental mode so have fun but expect issues. Requires `sageworks >= 0.8.60`. Feel free to submit issues to [SageWorks Github](https://github.com/SuperCowPowers/sageworks)

- Copy the example custom model script into your own directory
    - See: [Custom Model Script](https://github.com/SuperCowPowers/sageworks/tree/main/src/sageworks/model_scripts/custom_script_example)
- Make a requirements.txt and put into the same directory
- Train/deploy the ^existing^ example
    - This is an important step, don't skip it
    - If the existing model script trains/deploys your in great shape for the next step, if it doesn't then now is a good time to debug AWS account/permissions/etc.
- Now customize the model script
- Train/deploy your custom script

### Training/Deploying Custom Models
```python
from sageworks.api import ModelType
from sageworks.core.transforms.features_to_model.features_to_model import FeaturesToModel

# Note this directory should also have a requirements.txt in it
my_custom_script = "/full/path/to/my/directory/my_custom_script.py"
input_uuid = "wine_features"    # FeatureSet you want to use
output_uuid = "my-custom-model" # change to whatever
target_column = "wine-class"    # change to whatever
to_model = FeaturesToModel(input_uuid, output_uuid,
                           model_type=ModelType.CLASSIFIER, 
                           custom_script=my_custom_script)
to_model.set_output_tags(["your", "tags"])
to_model.transform(target_column=target_column, description="Custom Model")
``` 

### Custom Models: Create an Endpoint/Run Inference
```python
from sageworks.api import Model, Endpoint

model = Model("my-custom-model")
end = model.to_endpoint()   # Note: This takes a while

# Now run inference on my custom model :)
end.auto_inference(capture=True)

# Run inference with my own dataframe
df = fs.pull_dataframe()  # Or whatever dataframe
end.inference(df)
``` 

## Questions?
<img align="right" src="../../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and SageWorks. Please contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)