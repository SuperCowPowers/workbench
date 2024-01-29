# Meta

!!! tip inline end "Meta Examples"
    Examples of using the Meta class are listed at the bottom of this page [Examples](#examples).
    
::: sageworks.api.meta


## Examples
These example show how to use the `Meta()` class to pull lists of artifacts from AWS. DataSources, FeatureSets, Models, Endpoints and more. If you're building a web interface plugin, the **Meta** class is a great place to start.

!!!tip "SageWorks REPL"
    If you'd like to see **exactly** what data/details you get back from the `Meta()` class, you can spin up the SageWorks REPL, use the class and test out all the methods. Try it out! [SageWorks REPL](../repl/index.md)

```py title="Using SageWorks REPL"
[●●●]SageWorks:scp_sandbox> meta = Meta()
[●●●]SageWorks:scp_sandbox> model_info = meta.models()
[●●●]SageWorks:scp_sandbox> model_info
Out[5]:
{'wine-classification': [{'ModelPackageGroupName': 'wine-classification',
   'ModelPackageVersion': 1,
   'ModelPackageArn': 'arn:aws:sagemaker:us-west-2:123:model-package/wine-classification/1',
   'ModelPackageDescription': 'Wine Classification Model',
   ...
```

**List the Models in AWS**

```py title="meta_list_models.py"
from sageworks.api.meta import Meta

# Create our Meta Class and get a list of our Models
meta = Meta()
models = meta.models()

# Print out the list of our Models
model_list = list(models.keys())
print(f"Number of Models: {len(model_list)}")
for model_name in models.keys():
    print(f"\t{model_name}")
```

**Output**

```py
Number of Models: 2
	wine-classification
	abalone-regression
```

**Getting Model Performance Metrics**

```py title="meta_models.py"
from sageworks.api.meta import Meta

# Create our Meta Class and get a summary of our Models
meta = Meta()
models = meta.models()

# Print out the summary of our Models
for name, info in models.items():
    print(f"{name}")
    latest = info[0]  # We get a list of models, so we only want the latest
    print(f"\tARN: {latest['ModelPackageGroupArn']}")
    print(f"\tDescription: {latest['ModelPackageDescription']}")
    print(f"\tTags: {latest['sageworks_meta']['sageworks_tags']}")
    performance_metrics = latest["sageworks_meta"]["sageworks_inference_metrics"]
    print(f"\tPerformance Metrics:")
    for metric in performance_metrics.keys():
        print(f"\t\t{metric}: {performance_metrics[metric]}")
```

**Output**

```py
wine-classification
	ARN: arn:aws:sagemaker:us-west-2:123:model-package-group/wine-classification
	Description: Wine Classification Model
	Tags: wine:classification
	Performance Metrics:
		wine_class: {'0': 'TypeA', '1': 'TypeB', '2': 'TypeC'}
		precision: {'0': 1.0, '1': 1.0, '2': 1.0}
		recall: {'0': 1.0, '1': 1.0, '2': 1.0}
		fscore: {'0': 1.0, '1': 1.0, '2': 1.0}
		roc_auc: {'0': 1.0, '1': 1.0, '2': 1.0}
		support: {'0': 12, '1': 17, '2': 9}
abalone-regression
	ARN: arn:aws:sagemaker:us-west-2:123:model-package-group/abalone-regression
	Description: Abalone Regression Model
	Tags: abalone:regression
	Performance Metrics:
		MAE: {'0': 0.7705906241706162}
		RMSE: {'0': 1.2832401524233403}
		R2: {'0': 0.842409787754799}
		MAPE: {'0': 7.9788138931509085}
		MedAE: {'0': 0.4840479999999996}
```

**List the Endpoints in AWS**

```py title="meta_list_endpoints.py"
from sageworks.api.meta import Meta

# Create our Meta Class and get a list of our Endpoints
meta = Meta()
endpoints = meta.endpoints()

# Print out the list of our Endpoints
endpoint_list = list(endpoints.keys())
print(f"Number of Endpoints: {len(endpoint_list)}")
for name, info in endpoints.items():
    print(f"{name}")
    print(f"\tStatus: {info['EndpointStatus']}")
    print(f"\tInstance: {info['InstanceType']}")
```

**Output**

```py
Number of Endpoints: 2
wine-classification-end
	Status: InService
	Instance: Serverless (2GB/5)
abalone-regression-end
	Status: InService
	Instance: Serverless (2GB/5)
```


!!! note "Not Finding some particular AWS Data?"
    The SageWorks Meta API Classes uses the AWSServiceBroker and the 'Connector' Classes Internally, so for an extensive listing of all the methods available please take a deep dive into: [SageWorks Core Classes](../core_classes/overview.md)