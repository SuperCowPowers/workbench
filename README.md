
# Welcome to SageWorks
The SageWorks framework makes AWS® both easier to use and more powerful. SageWorks handles all the details around updating and managing a complex set of AWS Services. With a simple-to-use Python API and a beautiful set of web interfaces, SageWorks makes creating AWS ML pipelines a snap. It also dramatically improves both the usability and visibility across the entire spectrum of services: Glue Job, Athena, Feature Store, Models, and Endpoints, SageWorks makes it easy to build production ready, AWS powered, machine learning pipelines.

<img align="right" width="480" alt="sageworks_new_light" src="https://github.com/SuperCowPowers/sageworks/assets/4806709/ed2ed1bd-e2d8-49a1-b350-b2e19e2b7832">

### Full AWS ML OverView
- Health Monitoring 🟢
- Dynamic Updates
- High Level Summary

### Drill-Down Views
- Incoming Data
- Glue Jobs
- DataSources
- FeatureSets
- Models
- Endpoints

## Private SaaS Architecture
*Secure your Data, Empower your ML Pipelines*

SageWorks is architected as a **Private SaaS** (also called BYOC: Bring Your Own Cloud). This hybrid architecture is the ultimate solution for businesses that prioritize data control and security. SageWorks deploys as an AWS Stack within your own cloud environment, ensuring compliance with stringent corporate and regulatory standards. It offers the flexibility to tailor solutions to your specific business needs through our comprehensive plugin support, both components and full web interfaces. By using SageWorks, you maintain absolute control over your data while benefiting from the power, security, and scalability of AWS cloud services. [SageWorks Private SaaS Architecture](https://docs.google.com/presentation/d/1f_1gmE4-UAeUDDsoNdzK_d_MxALFXIkxORZwbJBjPq4/edit?usp=sharing)

<img alt="private_saas_compare" src="https://github.com/user-attachments/assets/2f6d3724-e340-4a70-bb97-d05383917cfe">

### API Installation

- ```pip install sageworks```  Installs SageWorks

- ```sageworks``` Runs the SageWorks REPL/Initial Setup

For the full instructions for connecting your AWS Account see:

- Getting Started: [Initial Setup](https://supercowpowers.github.io/sageworks/getting_started/) 
- One time AWS Onboarding: [AWS Setup](https://supercowpowers.github.io/sageworks/aws_setup/core_stack/)


### SageWorks Presentations
Even though SageWorks makes AWS easier, it's taking something very complex (Full AWS ML Pipelines/Services) and making it less complex. SageWorks has a depth and breadth of functionality so we've provided higher level conceptual documentation See: [SageWorks Presentations](https://supercowpowers.github.io/sageworks/presentations/)

### SageWorks Documentation
<img align="right" width="300" alt="sageworks_api" style="padding-left: 10px;"  src="https://github.com/SuperCowPowers/sageworks/assets/4806709/bf0e8591-75d4-44c1-be05-4bfdee4b7186">

The SageWorks documentation [SageWorks Docs](https://supercowpowers.github.io/sageworks/) covers our in-depth Python API and contains code examples. The code examples are provided in the Github repo `examples/` directory. For a full code listing of any example please visit our [SageWorks Examples](https://github.com/SuperCowPowers/sageworks/blob/main/examples)

### Questions?
The SuperCowPowers team is happy to anser any questions you may have about AWS and SageWorks. Please contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 


### SageWorks Beta Program
Using SageWorks will minimize the time and manpower needed to incorporate AWS ML into your organization. If your company would like to be a SageWorks Beta Tester, contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com).


### Using SageWorks with Additional Packages

```
pip install sageworks             # Installs SageWorks with Core Dependencies
pip install 'sageworks[ml-tools]' # + Shap and NetworkX
pip install 'sageworks[chem]'     # + RDKIT and Mordred (community)
pip install 'sageworks[ui]'       # + Plotly/Dash
pip install 'sageworks[dev]'      # + Pytest/flake8/black
pip install 'sageworks[all]'      # + All the things :)

*Note: Shells may interpret square brackets as globs, so the quotes are needed
```

### Contributions
If you'd like to contribute to the SageWorks project, you're more than welcome. All contributions will fall under the existing project [license](https://github.com/SuperCowPowers/sageworks/blob/main/LICENSE). If you are interested in contributing or have questions please feel free to contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com).

<img align="right" src="docs/images/scp.png" width="180">

® Amazon Web Services, AWS, the Powered by AWS logo, are trademarks of Amazon.com, Inc. or its affiliates
