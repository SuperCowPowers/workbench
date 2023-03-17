<img align="left" src="docs/images/scp.png" width="180">

# SageWorks<sup><i>TM</i></sup>

#### SageWorks: The scientist's workbench powered by AWS® for scalability, flexibility, and security.

SageWorks is a medium granularity framework that manages and aggregates AWS® Services into classes and concepts. When you use SageWorks you think about **DataSources**, **FeatureSets**, **Models**, and **Endpoints**. Underneath the hood those classes handle all the details around updating and managing a **complex set of AWS Services**. All the power and none of the pain so that your team can **Do Science Faster!**

<img src="docs/images/sageworks_concepts.png">

For more details see: [SageWorks Artitected FrameWork](https://docs.google.com/presentation/d/1ZiSy4ulEx5gfNQS76yRv8vgkehJ9gXRJ1PulutLKzis/edit?usp=sharing)


## Why SageWorks?

- The AWS SageMaker® ecosystem is **awesome** but has a large number of services with significant complexity
- SageWorks provides **rapid prototyping** through easy to use **classes** and **transforms**
- SageWorks provides **visibility** and **transparency** into AWS SageMaker® Pipelines
    - What S3 data sources are getting pulled?
    - What Features Store/Group is the Model Using?
    - What's the ***Provenance*** of a Model in Model Registry?
    - What SageMaker Endpoints are associated with this model?

<img src="docs/images/graph_representation.png" width="400">

<i><b> Clearly illustrated:</b> SageWorks uses Pipeline Graphs to provide intuitive and transparent visibility into AWS Sagemaker Deployments.</i>

## Installation
```
pip install sageworks
```

### SageWorks Zen
- The AWS SageMaker® set of services is vast and **complex**.
- SageWorks Classes encapsulate, organize, and manage sets of AWS® Services.
- **Heavy** transforms typically use **[AWS Athena](https://aws.amazon.com/athena/)** or **[Apache Spark](https://spark.apache.org/)** (AWS Glue/EMR Serverless).
- **Light** transforms will typically use **[Pandas](https://pandas.pydata.org/)**.
- Heavy and Light transforms both update **AWS Artifacts** (collections of AWS Services).
- **Quick prototypes** are typically built with the **light path** and then flipped to the **heavy path** as the system matures and usage grows.

### Classes and Concepts
The SageWorks Classes are orgnized to work in concert with AWS Services. For more details on the current classes and class heirarchies see [SageWorks Classes and Concepts](docs/sageworks_classes_concepts.md).

### Contributions
If you'd like to contribute to the SageWorks project, you're more than welcome. All contributions will fall under the existing project [license](https://github.com/SuperCowPowers/sageworks/blob/main/LICENSE). If you are interested in contributing or have questions please feel free to contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com).






### SageWorks Alpha Testers Wanted
Our experienced team can provide development and consulting services to help you effectively use Amazon’s Machine Learning services within your organization.

The popularity of cloud based Machine Learning services is booming. The problem many companies face is how that capability gets effectively used and harnessed to drive real business decisions and provide concrete value for their organization.

Using SageWorks will minimizize the time and manpower needed to incorporate AWS ML into your organization. If your company would like to be a SageWorks Alpha Tester, contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com).

® Amazon Web Services, AWS, the Powered by AWS logo, are trademarks of Amazon.com, Inc. or its affiliates.