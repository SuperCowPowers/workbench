# Meta

!!! tip inline end "Meta Examples"
    Examples of using the Meta class are listed at the bottom of this page [Examples](#examples).
    
::: sageworks.api.meta


## Examples
These example show how to use the `Meta()` class to pull lists of artifacts from AWS. DataSources, FeatureSets, Models, Endpoints and more. If you're building a web interface plugin, the **Meta** class is a great place to start.

!!!tip "SageWorks REPL"
    If you'd like to see **exactly** what data/details you get back from the `Meta()` class, you can spin up the SageWorks REPL, use the class and test out all the methods. Try it out! [SageWorks REPL](../repl/index.md)

```py title="Using SageWorks REPL"
meta = Meta()
model_df = meta.models()
model_df
               Model Group   Health Owner  ...             Input     Status                Description
0      wine-classification  healthy     -  ...     wine_features  Completed  Wine Classification Model
1  abalone-regression-full  healthy     -  ...  abalone_features  Completed   Abalone Regression Model
2       abalone-regression  healthy     -  ...  abalone_features  Completed   Abalone Regression Model

[3 rows x 10 columns]
```

**List the Models in AWS**

```py title="meta_list_models.py"
from sageworks.api import Meta

# Create our Meta Class and get a list of our Models
meta = Meta()
model_df = meta.models()

print(f"Number of Models: {len(model_df)}")
print(model_df)

# Get more details data on the Models
model_names = model_df["Model Group"].tolist()
for name in model_names:
    pprint(meta.model(name))
```

**Output**

```py
Number of Models: 3
               Model Group   Health Owner  ...             Input     Status                Description
0      wine-classification  healthy     -  ...     wine_features  Completed  Wine Classification Model
1  abalone-regression-full  healthy     -  ...  abalone_features  Completed   Abalone Regression Model
2       abalone-regression  healthy     -  ...  abalone_features  Completed   Abalone Regression Model

[3 rows x 10 columns]
wine-classification
abalone-regression-full
abalone-regression
```

**Getting Model Performance Metrics**

```py title="meta_model_metrics.py"
from sageworks.api import Meta

# Create our Meta Class and get a list of our Models
meta = Meta()
model_df = meta.models()

print(f"Number of Models: {len(model_df)}")
print(model_df)

# Get more details data on the Models
model_names = model_df["Model Group"].tolist()
for name in model_names[:5]:
    model_details = meta.model(name)
    print(f"\n\nModel: {name}")
    performance_metrics = model_details["sageworks_meta"]["sageworks_inference_metrics"]
    print(f"\tPerformance Metrics: {performance_metrics}")
```

**Output**

```py
wine-classification
	ARN: arn:aws:sagemaker:us-west-2:507740646243:model-package-group/wine-classification
	Description: Wine Classification Model
	Tags: wine::classification
	Performance Metrics:
		[{'wine_class': 'TypeA', 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0, 'roc_auc': 1.0, 'support': 12}, {'wine_class': 'TypeB', 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0, 'roc_auc': 1.0, 'support': 14}, {'wine_class': 'TypeC', 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0, 'roc_auc': 1.0, 'support': 9}]

abalone-regression
	ARN: arn:aws:sagemaker:us-west-2:507740646243:model-package-group/abalone-regression
	Description: Abalone Regression Model
	Tags: abalone::regression
	Performance Metrics:
		[{'MAE': 1.64, 'RMSE': 2.246, 'R2': 0.502, 'MAPE': 16.393, 'MedAE': 1.209, 'NumRows': 834}]
```

**List the Endpoints in AWS**

```py title="meta_list_endpoints.py"
from pprint import pprint
from sageworks.api import Meta

# Create our Meta Class and get a list of our Endpoints
meta = Meta()
endpoint_df = meta.endpoints()
print(f"Number of Endpoints: {len(endpoint_df)}")
print(endpoint_df)

# Get more details data on the Endpoints
endpoint_names = endpoint_df["Name"].tolist()
for name in endpoint_names:
    pprint(meta.endpoint(name))
```

**Output**

```py
Number of Endpoints: 2
                      Name   Health            Instance           Created  ...     Status     Variant Capture Samp(%)
0  wine-classification-end  healthy  Serverless (2GB/5)  2024-03-23 23:09  ...  InService  AllTraffic   False       -
1   abalone-regression-end  healthy  Serverless (2GB/5)  2024-03-23 21:11  ...  InService  AllTraffic   False       -

[2 rows x 10 columns]
wine-classification-end
<lots of details about endpoints>
```


!!! note "Not Finding some particular AWS Data?"
    The SageWorks Meta API Class also has `(details=True)` arguments, so make sure to check those out.
