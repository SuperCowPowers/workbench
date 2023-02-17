<img align="right" src="docs/images/sageworks.png" width="340">
<img align="left" src="docs/images/scp.png" width="180">

# SageWorks<sup><i>TM</i></sup>

AWS SageMaker has a fantastic set of functional components that can be used in concert to set up production level data processing and machine learning functionality.

- **Training Data:** Organized S3 buckets for training data
- **Feature Store:** Store/organize 'curated/known' feature sets
- **Model Registery:** Models with known performance stats/Model Scoreboards
- **Model Endpoints:** Easy to use HTTP(S) endpoints for single or batch predictions


## Why SageWorks?

- The Amazon SageMaker ecosystem has significant complexity
- SageWorks lets us set up ML Pipelines in a few lines of code
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

### SageWorks: Alpha Testers
Our experienced team can provide development and consulting services to help you effectively use Amazon’s Machine Learning services within your organization.

The popularity of cloud based Machine Learning services is booming. The problem many companies face is how that capability gets effectively used and harnessed to drive real business decisions and provide concrete value for their organization.

Using SageWorks will minimizize the time and manpower needed to incorporate AWS ML into your organization. If your company would like to be a SageWorks Alpha Tester, contact us at [sageworks@supercowpower.com](mailto:sageworks@supercowpower.com).

### Contributions
If you'd like to contribute to the SageWorks project, you're more than welcome. All contributions will fall under the existing project [licence](https://github.com/SuperCowPowers/sageworks/blob/main/LICENSE). If you are interested in contributing or have questions please feel free to contact us at [sageworks@supercowpower.com](mailto:sageworks@supercowpower.com).