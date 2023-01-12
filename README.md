<img align="right" style="padding: 0px 0px 20px 20px" src="docs/images/scp_labs.png" width="280">

# SageWorks<sup><i>TM</i></sup>
AWS SageMaker has a fantastic set of functional components that can be used in concert to setup production level data processing and machine learning functionality.

- **Training Data:** Organized S3 buckets for training data
- **Feature Store:** Store/organize 'curated/known' feature sets
- **Model Registery:** Models with known performance stats/Model Scoreboards
- **Model Endpoints:** Easy to use HTTP(S) endpoints for single or batch predictions


## Why SageWorks?
- SageMaker is awesome but fairly complex
- SageWorks lets us setup SageMaker Pipelines in a few lines of code
- Pipeline Graphs: Visibility/Transparency into a Pipeline
    - What S3 data sources are getting pulled?
    - What Features Store(s) is the Model Using?
    - What's the ***Provenance*** of a Model in Model Registry?
    - What SageMaker Endpoints are associated with this model?

    
## Installation
```
pip install sageworks
```
