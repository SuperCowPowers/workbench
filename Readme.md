<img align="left" src="docs/images/scp.png" width="180">

# SageWorks<sup><i>TM</i></sup>

#### SageWorks: The scientist's workbench powered by AWS® for scalability, flexibility, and security.

SageWorks is a medium granularity framework that manages and aggregates AWS® Services into classes and concepts. When you use SageWorks you think about **DataSources**, **FeatureSets**, **Models**, and **Endpoints**. Underneath the hood those classes handle all the details around updating and managing a **complex set of AWS Services**. All the power and none of the pain so that your team can **Do Science Faster!**

<img src="docs/images/sageworks_concepts.png">

### Full SageWorks OverView
[SageWorks Architected FrameWork](https://docs.google.com/presentation/d/1ZiSy4ulEx5gfNQS76yRv8vgkehJ9gXRJ1PulutLKzis/edit?usp=sharing)

### Getting Started Video Demo
[SageWorks Getting Started](https://drive.google.com/file/d/1iO7IuQtTYdx4BtQjxv9lI1aVJ2ZcAo43/view?usp=sharing) This video is an informal screen capture + chatting while I'm coding. The video demonstrates creating an entire AWS ML Pipeline, from data ingestion, to feature sets, to models and endpoints.



## Why SageWorks?

<img align="right" src="docs/images/graph_representation.png" width="300">

- The AWS SageMaker® ecosystem is **awesome** but has a large number of services with significant complexity
- SageWorks provides **rapid prototyping** through easy to use **classes** and **transforms**
- SageWorks provides **visibility** and **transparency** into AWS SageMaker® Pipelines
    - What S3 data sources are getting pulled?
    - What Features Store/Group is the Model Using?
    - What's the ***Provenance*** of a Model in Model Registry?
    - What SageMaker Endpoints are associated with this model?


**Single pane of glass** visibility into the AWS Services that underpin the SageWorks Classes. We can see that SageWorks automatically tags and tracks the inputs of all artifacts providing 'data provenance' for all steps in the AWS modeling pipeline.

<img width="1000" alt="Screenshot 2023-03-31 at 2 16 36 PM" src="https://user-images.githubusercontent.com/4806709/229222245-59e342c1-7254-47de-a453-268448643143.png">

<i><b> Clearly illustrated:</b> SageWorks provides intuitive and transparent visibility into the full pipeline of your AWS Sagemaker Deployments.</i>

## Installation
```
pip install sageworks
```

### Gettting Started
SageWorks has a large number of classes and components, the best place to get started is our [SageWorks Overview](https://docs.google.com/presentation/d/1ZiSy4ulEx5gfNQS76yRv8vgkehJ9gXRJ1PulutLKzis/edit?usp=sharing) and [SageWorks Wiki](https://github.com/SuperCowPowers/sageworks/wiki) for developers wanting to try out the Python API. 

Video: [SageWorks Getting Started](https://drive.google.com/file/d/1iO7IuQtTYdx4BtQjxv9lI1aVJ2ZcAo43/view?usp=sharing) shows an informal screen capture + chatting while I'm coding. The video demonstrates creating an entire AWS ML Pipeline, from data ingestion, to feature sets, to models and endpoints.

You can also join us on Discord <https://discord.gg/WHAJuz8sw8> for questions and advice on using SageWorks within your organization.

### SageWorks Zen``
- The AWS SageMaker® set of services is vast and **complex**.
- SageWorks Classes encapsulate, organize, and manage sets of AWS® Services.
- **Heavy** transforms typically use **[AWS Athena](https://aws.amazon.com/athena/)** or **[Apache Spark](https://spark.apache.org/)** (AWS Glue/EMR Serverless).
- **Light** transforms will typically use **[Pandas](https://pandas.pydata.org/)**.
- Heavy and Light transforms both update **AWS Artifacts** (collections of AWS Services).
- **Quick prototypes** are typically built with the **light path** and then flipped to the **heavy path** as the system matures and usage grows.

### Classes and Concepts
The SageWorks Classes are organized to work in concert with AWS Services. For more details on the current classes and class hierarchies see [SageWorks Classes and Concepts](docs/sageworks_classes_concepts.md).

### Contributions
If you'd like to contribute to the SageWorks project, you're more than welcome. All contributions will fall under the existing project [license](https://github.com/SuperCowPowers/sageworks/blob/main/LICENSE). If you are interested in contributing or have questions please feel free to contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com).


### SageWorks Alpha Testers Wanted
Our experienced team can provide development and consulting services to help you effectively use Amazon’s Machine Learning services within your organization.

The popularity of cloud based Machine Learning services is booming. The problem many companies face is how that capability gets effectively used and harnessed to drive real business decisions and provide concrete value for their organization.

Using SageWorks will minimizize the time and manpower needed to incorporate AWS ML into your organization. If your company would like to be a SageWorks Alpha Tester, contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com).

® Amazon Web Services, AWS, the Powered by AWS logo, are trademarks of Amazon.com, Inc. or its affiliates.
