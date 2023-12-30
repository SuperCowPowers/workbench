# Transforms
!!! tip inline end "API Classes"
    For most users the [API Classes](../../api_classes/overview.md) will provide all the general functionality to create a full AWS ML Pipeline

SageWorks currently has a large set of Transforms that go from one Artifact type to another (e.g. DataSource to FeatureSet). The Transforms will often have **light** and **heavy** versions depending on the scale of data that needs to be transformed.

## Transform Details

- **[DataLoaders Light](data_loaders_light.md):** Loads various light/smaller data into AWS Data Catalog and Athena
- **[DataLoaders Heavy](data_loaders_heavy.md):** Loads heavy/larger data (via Glue) into AWS Data Catalog and Athena
- **[DataToFeatures](data_to_features.md):** Transforms a DataSource into a FeatureSet (AWS Feature Store/Group)
- **[FeaturesToModel](features_to_model.md):** Trains and deploys an AWS Model Package/Group from a FeatureSet
- **[ModelToEndpoint](model_to_endpoint.md):** Manages the provisioning and deployment of a Model Endpoint
- **[PandasTransforms](pandas_transforms.md):** Pandas DataFrame transforms and helper methods.

![ML Pipeline](../../images/sageworks_concepts.png)