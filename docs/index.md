# Welcome to SageWorks
The SageWorks framework makes AWS® both easier to use and more powerful. SageWorks handles all the details around updating and managing a complex set of AWS Services. With a simple-to-use Python API and a beautiful set of web interfaces, SageWorks makes creating AWS ML pipelines a snap. It also dramatically improves both the usability and visibility across the entire spectrum of services: Glue Jobs, Athena, Feature Store, Models, and Endpoints. SageWorks makes it easy to build production ready, AWS powered, machine learning pipelines.

<figure style="float: right; width: 500px;">
<img alt="sageworks_new_light" src="https://github.com/SuperCowPowers/sageworks/assets/4806709/5f8b32a2-ed72-45f2-bd96-91b7bbbccff4">
<figcaption>SageWorks Dashboard: AWS Pipelines in a Whole New Light!</figcaption>
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


## Getting Started

The SageWorks package has two main components, a Web Interface that provides visibility into AWS ML PIpelines and a Python API that makes creation and usage of the AWS ML Services easier than using/learning the services directly.

### Web Interfaces
The SageWorks Dashboard has a set of web interfaces that give visibility into the AWS Glue and SageMaker Services. There are currently 5 web interfaces available:


- **Top Level Dashboard:** Shows all AWS ML Artifacts (Glue and SageMaker)
- **DataSources:** DataSource Column Details, Distributions and Correlations
- **FeatureSets:** FeatureSet Details, Distributions and Correlations
- **Model:** Model details, performance metric, and inference plots
- **Endpoints:** Endpoint details, realtime charts of endpoint performance and latency

### Python API
SageWorks API Documentation: [SageWorks API Classes](api_classes/overview.md) 

The main functionality of the Python API is to encapsulate and manage a set of AWS services underneath a Python Object interface. The Python Classes are used to create and interact with Machine Learning Pipeline Artifacts.

### Initial Setup/Config
**Note:** Use the SageWorks REPL to setup your AWS connection for both API Usage (Data Scientists/Engineers) and AWS Initial Setup (AWS Folks).

```
> pip install sageworks
> sageworks <-- This starts the REPL

Welcome to SageWorks!
Looks like this is your first time using SageWorks...
Let's get you set up...
AWS_PROFILE: my_aws_profile
SAGEWORKS_BUCKET: my-company-sageworks
[optional] REDIS_HOST(localhost): my-redis.cache.amazon (or leave blank)
[optional] REDIS_PORT(6379):
[optional] REDIS_PASSWORD():
[optional] SAGEWORKS_API_KEY(open_source): my_api_key (or leave blank)
```
**That's It:** You're now all set. This configuration only needs to be **ONCE** :)

### AWS Folks (initial setup)
Setting up SageWorks on your AWS Account: [AWS Setup](aws_setup/core_stack.md)

### Data Scientists/Engineers
- SageWorks REPL: [SageWorks REPL](repl/index.md)
- Using SageWorks for ML Pipelines: [SageWorks API Classes](api_classes/overview.md)
- SCP SageWorks Github: [Github Repo](https://github.com/SuperCowPowers/sageworks)


## Additional Resources

<img align="right" src="images/scp.png" width="180">

- SageWorks Core Classes: [Core Classes](core_classes/overview.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
