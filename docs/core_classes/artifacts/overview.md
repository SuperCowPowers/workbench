# SageWorks Artifacts
!!! tip inline end "API Classes"
    For most users the [API Classes](../../api_classes/overview.md) will provide all the general functionality to create a full AWS ML Pipeline

## Welcome to the SageWorks Core Artifact Classes

These classes provide low-level APIs for the SageWorks package, they interact more directly with AWS Services and are therefore more complex with a fairly large number of methods. 

- **[AthenaSource](athena_source.md):** Manages AWS Data Catalog and Athena
- **[FeatureSetCore](feature_set_core.md):** Manages AWS Feature Store and Feature Groups
- **[ModelCore](model_core.md):** Manages the training and deployment of AWS Model Groups and Packages
- **[EndpointCore](endpoint_core.md):** Manages the deployment and invocations/inference on AWS Endpoints

![ML Pipeline](../../images/sageworks_concepts.png)
