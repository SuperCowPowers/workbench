# SageWorks Classes and Concepts
A flexible, rapid, and customizable AWS® ML Sandbox. Here's some of the classes and concepts we use in the SageWorks system:

**Note:** If you're currently developing an application against the SageWorks Classes/APIs please note that the API is currently in a fairly fluid state right now. :)

- Artifacts
  - DataLoader
  - DataSource
  - FeatureSet
  - Model
  - Endpoint
- Transforms (Script, Glue, Lambda)
  - DataSource to DataSource (Light/Heavy)
  - DataSource to FeatureSet (Light/Heavy)
  - FeatureSet to FeatureSet (Light/Heavy)
  - FeatureSet to Model
  - Model to Endpoint 
- Glue/Heavy Lifting
  - All transorms can be AWS Spark/Glue Jobs
  - Easily handle the biggest workloads
- Lambdas
  - All transorms can be Lambdas
  - Lambdas have formal inputs/outputs	

  Example Transform Classes:
  
  	 - MyDataToFeatures: **DataSource** --> **FeatureSet**
  	 - RemoveCorrelated: **FeatureSet** --> **FeatureSet**
  	 - TrainModel: **FeatureSet** --> **Model**
  	 - ServeModel: **Model** --> **Endpoint** 


- Pipelines/DAGs of Tranforms (Step Functions)
  - WireUp Lambdas with matching input/output
  - Easy to build flexible and fantastic AWS ML Systems!