# SageWorks Classes and Concepts
A flexible, rapid, and customizable AWS® ML Sandbox. Here's some of the classes and concepts we use in the SageWorks system:

- Artifacts
  - DataLoader
  - DataSource
  - FeatureSet
  - Model
  - Endpoint
- Transforms on Artifacts (Script, Glue, or Lambdas)
  - DataSource to DataSource 
  - DataSource to FeatureSet 
  - FeatureSet to FeatureSet 
  - FeatureSet to Model
  - Model to Endpoint 
- Glue/Heavy Lifting
  - All transorms can be AWS Spark/Glue Jobs
  - Easily handle the biggest workloads
- Lambdas
  - All transorms can be Lambdas
  - Lambdas have formal inputs/outputs	
  	 - MyDataToFeatures: **DataSource** --> **FeatureSet**
  	 - RemoveCorrelated: **FeatureSet** --> **FeatureSet**
  	 - TrainModel: **FeatureSet** --> **Model**
  	 - ServeModel: **Model** --> **Endpoint** 


- Pipelines/DAGs of Tranforms (Step Functions)
  - WireUp Lambdas with matching input/output
  - Easy to build flexible and fantastic AWS ML Systems!