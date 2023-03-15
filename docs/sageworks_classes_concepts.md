# SageWorks Classes and Concepts
A flexible, rapid, and customizable AWS® ML Sandbox. Here's some of the classes and concepts we use in the SageWorks system:

<img src="images/sageworks_concepts.png">

- Artifacts
  - DataLoader
  - DataSource
  - FeatureSet
  - Model
  - Endpoint

- Transforms
  - DataSource to DataSource
     - Heavy 
         - AWS Glue Jobs
         - AWS EMR Serverless
     - Light
         - Local/Laptop
         - Lambdas
         - StepFunctions
  - DataSource to FeatureSet
     - Heavy/Light (see above breakout)
  - FeatureSet to FeatureSet
     - Heavy/Light (see above breakout)
  - FeatureSet to Model
  - Model to Endpoint

     
