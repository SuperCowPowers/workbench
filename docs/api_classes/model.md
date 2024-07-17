# Model

!!! tip inline end "Model Examples"
    Examples of using the Model Class are in the [Examples](#examples) section at the bottom of this page. AWS Model setup and deployment are quite complicated to do manually but the SageWorks Model Class makes it a breeze!

::: sageworks.api.model


## Examples
All of the SageWorks Examples are in the Sageworks Repository under the `examples/` directory. For a full code listing of any example please visit our [SageWorks Examples](https://github.com/SuperCowPowers/sageworks/blob/main/examples)

**Create a Model from a FeatureSet**

```py title="featureset_to_model.py"
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import ModelType
from pprint import pprint

# Grab a FeatureSet
my_features = FeatureSet("test_features")

# Create a Model from the FeatureSet
# Note: ModelTypes can be CLASSIFIER, REGRESSOR (XGBoost is default)
my_model = my_features.to_model(model_type=ModelType.REGRESSOR, 
                                target_column="iq_score")
pprint(my_model.details())

```

**Output**

```py
{'approval_status': 'Approved',
 'content_types': ['text/csv'],
 ...
 'inference_types': ['ml.t2.medium'],
 'input': 'test_features',
 'model_metrics':   metric_name  value
				0        RMSE  7.924
				1         MAE  6.554,
				2          R2  0.604,
 'regression_predictions':       iq_score  prediction
							0   136.519012  139.964460
							1   133.616974  130.819950
							2   122.495415  124.967834
							3   133.279510  121.010284
							4   127.881073  113.825005
    ...
 'response_types': ['text/csv'],
 'sageworks_tags': ['test-model'],
 'shapley_values': None,
 'size': 0.0,
 'status': 'Completed',
 'transform_types': ['ml.m5.large'],
 'uuid': 'test-model',
 'version': 1}
```

**Use a specific Scikit-Learn Model**

```py title="featureset_to_knn.py"
from sageworks.api.feature_set import FeatureSet
from pprint import pprint

# Grab a FeatureSet
my_features = FeatureSet("abalone_features")

# Transform FeatureSet into KNN Regression Model
# Note: model_class can be any sckit-learn model 
#  "KNeighborsRegressor", "BayesianRidge",
#  "GaussianNB", "AdaBoostClassifier", etc
my_model = my_features.to_model(
    model_class="KNeighborsRegressor",
    target_column="class_number_of_rings",
    name="abalone-knn-reg",
    description="Abalone KNN Regression",
    tags=["abalone", "knn"],
    train_all_data=True,
)
pprint(my_model.details())
```
**Another Scikit-Learn Example**

```py title="featureset_to_rfc.py"
from sageworks.api.feature_set import FeatureSet
from pprint import pprint

# Grab a FeatureSet
my_features = FeatureSet("wine_features")

# Using a Scikit-Learn Model
# Note: model_class can be any sckit-learn model ("KNeighborsRegressor", "BayesianRidge",
#       "GaussianNB", "AdaBoostClassifier", "Ridge, "Lasso", "SVC", "SVR", etc...)
my_model = my_features.to_model(
    model_class="RandomForestClassifier",
    target_column="wine_class",
    name="wine-rfc-class",
    description="Wine RandomForest Classification",
    tags=["wine", "rfc"]
)
pprint(my_model.details())
```

**Create an Endpoint from a Model**

!!! warning inline end "Endpoint Costs"
    Serverless endpoints are a great option, they have no AWS charges when not running. A **realtime** endpoint has less latency (no cold start) but AWS charges an hourly fee which can add up quickly!

```py title="model_to_endpoint.py"
from sageworks.api.model import Model

# Grab the abalone regression Model
model = Model("abalone-regression")

# By default, an Endpoint is serverless, you can
# make a realtime endpoint with serverless=False
model.to_endpoint(name="abalone-regression-end",
                  tags=["abalone", "regression"],
                  serverless=True)
```

**Model Health Check and Metrics**

```py title="model_metrics.py"
from sageworks.api.model import Model

# Grab the abalone-regression Model
model = Model("abalone-regression")

# Perform a health check on the model
# Note: The health_check() method returns 'issues' if there are any
#       problems, so if there are no issues, the model is healthy
health_issues = model.health_check()
if not health_issues:
    print("Model is Healthy")
else:
    print("Model has issues")
    print(health_issues)

# Get the model metrics and regression predictions
print(model.model_metrics())
print(model.regression_predictions())
```

**Output**

```py
Model is Healthy
  metric_name  value
0        RMSE  2.190
1         MAE  1.544
2          R2  0.504

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
```

## SageWorks UI
Running these few lines of code creates an AWS Model Package Group and an AWS Model Package. These model artifacts can be viewed in the Sagemaker Console/Notebook interfaces or in the SageWorks Dashboard UI.

<figure>
<img alt="sageworks_new_light" src="https://github.com/SuperCowPowers/sageworks/assets/4806709/0c5cc2f8-bcc2-406d-a66e-32dcbad0cc25">
<figcaption>SageWorks Dashboard: Models</figcaption>
</figure>


!!! note "Not Finding a particular method?"
    The SageWorks API Classes use the 'Core' Classes Internally, so for an extensive listing of all the methods available please take a deep dive into: [SageWorks Core Classes](../core_classes/overview.md)
