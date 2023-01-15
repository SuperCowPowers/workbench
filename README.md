<img align="right" src="docs/images/sageworks.png" width="340">
<img align="left" src="docs/images/scp.png" width="180">

# SageWorks<sup><i>TM</i></sup>

AWS SageMaker has a fantastic set of functional components that can be used in concert to setup production level data processing and machine learning functionality.

- **Training Data:** Organized S3 buckets for training data
- **Feature Store:** Store/organize 'curated/known' feature sets
- **Model Registery:** Models with known performance stats/Model Scoreboards
- **Model Endpoints:** Easy to use HTTP(S) endpoints for single or batch predictions


## Why SageWorks?

- The Amazon SageMaker ecosystem has significant complexity
- SageWorks lets us setup ML Pipelines in a few lines of code
- **Pipeline Graphs:** Visibility/Transparency into our Pipelines
    - What S3 data sources are getting pulled?
    - What Features Store(s) is the Model Using?
    - What's the ***Provenance*** of a Model in Model Registry?
    - What SageMaker Endpoints are associated with this model?

<img src="docs/images/graph_representation.png" width="400">

<i><b> Clearly illustrated:</b> SageWorks uses Pipeline Graphs to provide intuitive and transparent visibility into AWS Sagemaker Deployments.</i>


    
## Installation
```
pip install sageworks
```
