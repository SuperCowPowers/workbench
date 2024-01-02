# Core Classes

!!! warning inline end "SageWorks Core Classes"
    These classes interact with many of the AWS service details and are therefore more complex. They provide additional control and refinement over the AWS ML Pipline. For most use cases the [API Classes](../api_classes/overview.md) should be used

**Welcome to the SageWorks Core Classes**

The Core Classes provide low-level APIs for the SageWorks package, these classes directly interface with the AWS Sagemaker Pipeline interfaces and have a large number of methods with reasonable complexity.

The [API Classes](../api_classes/overview.md) have method pass-through so just call the method on the API Class and voilà it works the same.

![ML Pipeline](../images/sageworks_concepts.png)

## Artifacts
- **[AthenaSource](artifacts/athena_source.md):** Manages AWS Data Catalog and Athena
- **[FeatureSetCore](artifacts/feature_set_core.md):** Manages AWS Feature Store and Feature Groups
- **[ModelCore](artifacts/model_core.md):** Manages the training and deployment of AWS Model Groups and Packages
- **[EndpointCore](artifacts/endpoint_core.md):** Manages the deployment and invocations/inference on AWS Endpoints

## Transforms
Transforms are a set of classes that **transform** one type of `Artifact` to another type. For instance `DataToFeatureSet` takes a `DataSource` artifact and creates a `FeatureSet` artifact.

- **[DataLoaders Light](transforms/data_loaders_light.md):** Loads various light/smaller data into AWS Data Catalog and Athena
- **[DataLoaders Heavy](transforms/data_loaders_heavy.md):** Loads heavy/larger data (via Glue) into AWS Data Catalog and Athena
- **[DataToFeatures](transforms/data_to_features.md):** Transforms a DataSource into a FeatureSet (AWS Feature Store/Group)
- **[FeaturesToModel](transforms/features_to_model.md):** Trains and deploys an AWS Model Package/Group from a FeatureSet
- **[ModelToEndpoint](transforms/model_to_endpoint.md):** Manages the provisioning and deployment of a Model Endpoint
- **[PandasTransforms](transforms/pandas_transforms.md):**Pandas DataFrame transforms and helper methods.



