
# Recent News

### Themes

Everyone knows that good data science requires... **Some Awesome Themes!**

<table>
  <tr>
    <td>
      <a href="https://github.com/user-attachments/assets/82ab4eab-0688-4b93-ad8e-9b954564777b">
        <img width="400" alt="theme_dark" src="https://github.com/user-attachments/assets/82ab4eab-0688-4b93-ad8e-9b954564777b" />
      </a>
    </td>
    <td>
      <a href="https://github.com/user-attachments/assets/b63a0789-c144-4048-afb6-f03e3d993680">
        <img width="400" alt="theme_light" src="https://github.com/user-attachments/assets/b63a0789-c144-4048-afb6-f03e3d993680" />
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/user-attachments/assets/8a59be19-0c5d-42c6-9922-feafb1a1eecd">
        <img width="400" alt="theme_quartz" src="https://github.com/user-attachments/assets/8a59be19-0c5d-42c6-9922-feafb1a1eecd" />
      </a>
    </td>
    <td>
      <a href="https://github.com/user-attachments/assets/5b01ec64-8d56-43bf-96c5-7da8ec48f527">
        <img width="400" alt="theme_quartz_dark" src="https://github.com/user-attachments/assets/5b01ec64-8d56-43bf-96c5-7da8ec48f527" />
      </a>
    </td>
  </tr>
</table>

All of the Dashboard pages, subpages, and plugins use our new `ThemeManager()` class. See [Workbench Themes](https://supercowpowers.github.io/workbench/themes/), also big thanks to our friends at [Dash Bootstrap Templates](https://github.com/AnnMarieW/dash-bootstrap-templates)



### Workbench up on the AWS Marketplace

Powered by AWS® to accelerate your Machine Learning Pipelines development with our new [Dashboard for ML Pipelines](https://aws.amazon.com/marketplace/pp/prodview-5idedc7uptbqo). Getting started with Workbench is a snap and can be billed through AWS.

**Road Map: `v0.9.0`**

We've used the feedback from our current beta testers to improve the framework and we've constructed a mini road map for the upcoming Workbench version 0.9.0. Please see [Workbench RoadMaps](https://supercowpowers.github.io/workbench/road_maps/0_9_0/) 

# Welcome to Workbench
The Workbench framework makes AWS® both easier to use and more powerful. Workbench handles all the details around updating and managing a complex set of AWS Services. With a simple-to-use Python API and a beautiful set of web interfaces, Workbench makes creating AWS ML pipelines a snap. It also dramatically improves both the usability and visibility across the entire spectrum of services: Glue Job, Athena, Feature Store, Models, and Endpoints, Workbench makes it easy to build production ready, AWS powered, machine learning pipelines.

<img align="right" width="480" alt="workbench_new_light" src="https://github.com/SuperCowPowers/workbench/assets/4806709/ed2ed1bd-e2d8-49a1-b350-b2e19e2b7832">

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

Workbench is architected as a **Private SaaS** (also called BYOC: Bring Your Own Cloud). This hybrid architecture is the ultimate solution for businesses that prioritize data control and security. Workbench deploys as an AWS Stack within your own cloud environment, ensuring compliance with stringent corporate and regulatory standards. It offers the flexibility to tailor solutions to your specific business needs through our comprehensive plugin support. By using Workbench, you maintain absolute control over your data while benefiting from the power, security, and scalability of AWS cloud services. [Workbench Private SaaS Architecture](https://docs.google.com/presentation/d/1f_1gmE4-UAeUDDsoNdzK_d_MxALFXIkxORZwbJBjPq4/edit?usp=sharing)

<img alt="private_saas_compare" src="https://github.com/user-attachments/assets/2f6d3724-e340-4a70-bb97-d05383917cfe">

### API Installation

- ```pip install workbench```  Installs Workbench

- ```workbench``` Runs the Workbench REPL/Initial Setup

For the full instructions for connecting your AWS Account see:

- Getting Started: [Initial Setup](https://supercowpowers.github.io/workbench/getting_started/) 
- One time AWS Onboarding: [AWS Setup](https://supercowpowers.github.io/workbench/aws_setup/core_stack/)


### Workbench Presentations
Even though Workbench makes AWS easier, it's taking something very complex (the full set of AWS ML Pipelines/Services) and making it less complex. Workbench has a depth and breadth of functionality so we've provided higher level conceptual documentation See: [Workbench Presentations](https://supercowpowers.github.io/workbench/presentations/)

<img align="right" width="420" alt="workbench_api" style="padding-left: 10px;"  src="https://github.com/SuperCowPowers/workbench/assets/4806709/bf0e8591-75d4-44c1-be05-4bfdee4b7186">

### Workbench Documentation

The Workbench documentation [Workbench Docs](https://supercowpowers.github.io/workbench/) covers the Python API in depth and contains code examples. The documentation is fully searchable and fairly comprehensive.

The code examples are provided in the Github repo `examples/` directory. For a full code listing of any example please visit our [Workbench Examples](https://github.com/SuperCowPowers/workbench/blob/main/examples)

## Questions?
The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 


### Workbench Beta Program
Using Workbench will minimize the time and manpower needed to incorporate AWS ML into your organization. If your company would like to be a Workbench Beta Tester, contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com).


### Using Workbench with Additional Packages

```
pip install workbench             # Installs Workbench with Core Dependencies
pip install 'workbench[ml-tools]' # + Shap and NetworkX
pip install 'workbench[ui]'       # + Plotly/Dash
pip install 'workbench[dev]'      # + Pytest/flake8/black
pip install 'workbench[all]'      # + All the things :)

*Note: Shells may interpret square brackets as globs, so the quotes are needed
```

### Contributions
If you'd like to contribute to the Workbench project, you're more than welcome. All contributions will fall under the existing project [license](https://github.com/SuperCowPowers/workbench/blob/main/LICENSE). If you are interested in contributing or have questions please feel free to contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com).

<img align="right" src="docs/images/scp.png" width="180">

® Amazon Web Services, AWS, the Powered by AWS logo, are trademarks of Amazon.com, Inc. or its affiliates
