<img align="left" src="docs/images/scp.png" width="180">

# SageWorks<sup><i>TM</i></sup>

<img src="docs/images/sageworks_concepts.png" width="1000">


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

Using SageWorks will minimizize the time and manpower needed to incorporate AWS ML into your organization. If your company would like to be a SageWorks Alpha Tester, contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com).

### Contributions
If you'd like to contribute to the SageWorks project, you're more than welcome. All contributions will fall under the existing project [license](https://github.com/SuperCowPowers/sageworks/blob/main/LICENSE). If you are interested in contributing or have questions please feel free to contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com).

### SageWorks Zen
- [**AWS Glue**](https://docs.aws.amazon.com/glue/latest/dg/how-it-works.html) should maybe be called '**AWS Spark**'
- **Heavy** components typically use **[Apache Spark](https://spark.apache.org/)** (via AWS Glue/Spark).
- **Light** components will typically use **[Pandas](https://pandas.pydata.org/)**.
- **Heavy** and **Light** paths lead to **same** data/artifacts pushed in AWS Services.
- **Quick prototypes** can be built with the **light path** and then flipped to **heavy** as the system matures and usage grows.

### SageWorks Algorithm Zen
- There's only **trees**, well except for when it's a **graph** or a **DAG** or an **unordered collection**, but mostly it's **trees** .