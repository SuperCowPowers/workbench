# Welcome to Workbench
The Workbench framework makes AWS® both easier to use and more powerful. Workbench handles all the details around updating and managing a complex set of AWS Services. With a simple-to-use Python API and a beautiful set of web interfaces, Workbench makes creating AWS ML pipelines a snap. It also dramatically improves both the usability and visibility across the entire spectrum of services: Glue Jobs, Athena, Feature Store, Models, and Endpoints. Workbench makes it easy to build production ready, AWS powered, machine learning pipelines.

<figure style="float: right; width: 500px;">
<img alt="workbench_new_light" src="https://github.com/SuperCowPowers/workbench/assets/4806709/5f8b32a2-ed72-45f2-bd96-91b7bbbccff4">
<figcaption>Workbench Dashboard: AWS Pipelines in a Whole New Light!</figcaption>
</figure>

### Full AWS OverView
- Health Monitoring 🟢
- Dynamic Updates
- High Level Summary

### Drill-Down Views
- Glue Jobs
- DataSources
- FeatureSets
- Models
- Endpoints

## Private SaaS Architecture
**Secure your Data, Empower your ML Pipelines**

Workbench is architected as a Private SaaS. This hybrid architecture is the ultimate solution for businesses that prioritize data control and security. Workbench deploys as an AWS Stack within your own cloud environment, ensuring compliance with stringent corporate and regulatory standards. It offers the flexibility to tailor solutions to your specific business needs through our comprehensive plugin support, both components and full web interfaces. By using Workbench, you maintain absolute control over your data while benefiting from the power, security, and scalability of AWS cloud services. [Workbench Private SaaS Architecture](https://docs.google.com/presentation/d/1f_1gmE4-UAeUDDsoNdzK_d_MxALFXIkxORZwbJBjPq4/edit?usp=sharing)

## Dashboard and API

The Workbench package has two main components, a Web Interface that provides visibility into AWS ML PIpelines and a Python API that makes creation and usage of the AWS ML Services easier than using/learning the services directly.

### Web Interfaces
The Workbench Dashboard has a set of web interfaces that give visibility into the AWS Glue and SageMaker Services. There are currently 5 web interfaces available:


- **Top Level Dashboard:** Shows all AWS ML Artifacts (Glue and SageMaker)
- **DataSources:** DataSource Column Details, Distributions and Correlations
- **FeatureSets:** FeatureSet Details, Distributions and Correlations
- **Model:** Model details, performance metric, and inference plots
- **Endpoints:** Endpoint details, realtime charts of endpoint performance and latency

### Python API
Workbench API Documentation: [Workbench API Classes](api_classes/overview.md) 

The main functionality of the Python API is to encapsulate and manage a set of AWS services underneath a Python Object interface. The Python Classes are used to create and interact with Machine Learning Pipeline Artifacts.

## Getting Started
Workbench will need some initial setup when you first start using it. See our [Getting Started](getting_started/index.md) guide on how to connect Workbench to your AWS Account.


## Additional Resources

<img align="right" src="images/scp.png" width="180">

- Getting Started: [Getting Started](getting_started/index.md) 
- Workbench API Classes: [API Classes](api_classes/overview.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
