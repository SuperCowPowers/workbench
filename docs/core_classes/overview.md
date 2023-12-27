# Core Classes

!!! warning inline end "SageWorks Core Classes"
    These classes interact with many of the AWS service details and are therefore more complex. They try to balance usablility while maintaining a level of control and refinement over the AWS ML Pipline.

Welcome to the SageWorks Core Classes

The Core Classes provide low-level APIs for the SageWorks package, these classes directly interface with the AWS Sagemaker Pipeline interfaces and have a large number of methods with reasonable complexity.

 
    
For most users the [API Classes](../api_classes/overview.md) should provide the general functionality of creating a full AWS ML Pipeline but if you need more control feel free to investigate and use the Core Classes.

## Artifacts
Artifacts classes provide object oriented interfaces to sets of AWS Services. For instance the [FeatureSetCore][sageworks.core.artifacts.FeatureSetCore] encapsulates and interacts with AWS Feature Store, Feature Groups, Athena, and Data Catalogs.

For a full overview of the Artifact Core Class see
[Artifact Classes](artifacts/artifacts.md)

## Transforms
Transforms are a set of classes that **transform** one type of `Artifact` to another type. For instance `DataToFeatureSet` takes a `DataSource` artifact and creates a `FeatureSet` artifact.

For a full overview of the Transform Core Class see
[Transform Classes](transforms/overview.md)




