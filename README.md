
## Live Dashboard Demo
You can explore a live demo of the Workbench Dashboard at: [Workbench Dashboard Demo](https://workbench-dashboard.com)

## Recent News
**Chemprop Models!** All the rage for the Open ADMET Challenge. 

ADMET Workbench now supports:
- Single Task Chemprop Models
- Multi Task Chemprop Models
- Chemprop Hybrid Models (MPNN + Descriptors)
- Foundation Chemprop Models (CheMeleon Pretrained)

Examples: 

- [Deploying Chemprop Models](examples/models/chemprop.py)
- [Deploying Foundation Chemprop Models](examples/models/chemprop_foundation.py)

**References**
- [Open ADMET Challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)
- **ChemProp:** Yang et al. "Analyzing Learned Molecular Representations for Property Prediction" *J. Chem. Inf. Model.* 2019 — [GitHub](https://github.com/chemprop/chemprop) | [Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237)
- [CheMeleon Github](https://github.com/JacksonBurns/chemeleon)

### Chemprop Action Shots!

<table>
  <tr>
    <td>
      <a href="https://github.com/user-attachments/assets/a36c6eff-c464-4c9a-9859-a45cd7e35145">
        <img width="800" alt="theme_dark" src="https://github.com/user-attachments/assets/a36c6eff-c464-4c9a-9859-a45cd7e35145" />
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/user-attachments/assets/d65ec1da-e04e-44fe-8782-4da0fb50588a">
        <img width="800" alt="theme_quartz" src="https://github.com/user-attachments/assets/d65ec1da-e04e-44fe-8782-4da0fb50588a" />
      </a>
    </td>
  </tr>
</table>



# Welcome to ADMET Workbench
The ADMET Workbench framework makes AWS® both easier to use and more powerful. Workbench handles all the details around updating and managing a complex set of AWS Services. With a simple-to-use Python API and a beautiful set of web interfaces, Workbench makes creating AWS ML pipelines a snap. It also dramatically improves both the usability and visibility across the entire spectrum of services: Glue Job, Athena, Feature Store, Models, and Endpoints, Workbench makes it easy to build production ready, AWS powered, machine learning pipelines.

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

ADMET Workbench is architected as a **Private SaaS** (also called BYOC: Bring Your Own Cloud). This hybrid architecture is the ultimate solution for businesses that prioritize data control and security. Workbench deploys as an AWS Stack within your own cloud environment, ensuring compliance with stringent corporate and regulatory standards. It offers the flexibility to tailor solutions to your specific business needs through our comprehensive plugin support. By using Workbench, you maintain absolute control over your data while benefiting from the power, security, and scalability of AWS cloud services. [Workbench Private SaaS Architecture](https://docs.google.com/presentation/d/1f_1gmE4-UAeUDDsoNdzK_d_MxALFXIkxORZwbJBjPq4/edit?usp=sharing)

<img alt="private_saas_compare" src="https://github.com/user-attachments/assets/2f6d3724-e340-4a70-bb97-d05383917cfe">

### API Installation

For typical use (the API, REPL, dashboard, training pipelines):

- ```pip install 'workbench[all]'```  Full install — recommended
- ```workbench```                     Runs the Workbench REPL / initial setup

`pip install workbench` (no extras) is intentionally lightweight — it's the
endpoint-safe surface that ships inside SageMaker inference containers (and
the lambdas / scripts that just need to invoke endpoints). See
[Installation extras](#installation-extras) below for the breakdown.

For the full instructions for connecting your AWS Account see:

- Getting Started: [Initial Setup](https://supercowpowers.github.io/workbench/getting_started/) 
- One time AWS Onboarding: [AWS Setup](https://supercowpowers.github.io/workbench/aws_setup/core_stack/)


### ADMET Workbench up on the AWS Marketplace

Powered by AWS® to accelerate your Machine Learning Pipelines development with our new [Dashboard for ML Pipelines](https://aws.amazon.com/marketplace/pp/prodview-5idedc7uptbqo). Getting started with Workbench is a snap and can be billed through AWS.

### ADMET Workbench Presentations
Even though ADMET Workbench makes AWS easier, it's taking something very complex (the full set of AWS ML Pipelines/Services) and making it less complex. Workbench has a depth and breadth of functionality so we've provided higher level conceptual documentation See: [Workbench Presentations](https://supercowpowers.github.io/workbench/presentations/)

<img align="right" width="420" alt="workbench_api" style="padding-left: 10px;"  src="https://github.com/SuperCowPowers/workbench/assets/4806709/bf0e8591-75d4-44c1-be05-4bfdee4b7186">

### ADMET Workbench Documentation

The ADMET Workbench documentation [Workbench Docs](https://supercowpowers.github.io/workbench/) covers the Python API in depth and contains code examples. The documentation is fully searchable and fairly comprehensive.

The code examples are provided in the Github repo `examples/` directory. For a full code listing of any example please visit our [Workbench Examples](https://github.com/SuperCowPowers/workbench/blob/main/examples)

## Questions?
The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 


### ADMET Workbench Beta Program
Using ADMET Workbench will minimize the time and manpower needed to incorporate AWS ML into your organization. If your company would like to be a Workbench Beta Tester, contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com).


### Installation extras

Workbench's dependencies are organized so you can install exactly what you
need. The `workbench.endpoints.*` surface is enforced by a CI smoke test
that runs the lightweight install in a clean venv and verifies every module
under that namespace imports without any extras pulled in — which is what
makes the base install safe to drop into a SageMaker endpoint container or
a lambda.

```
pip install workbench               # Endpoint-safe core only:
                                    #   boto3, awswrangler, numpy, pandas,
                                    #   sklearn, scipy, rdkit, joblib
                                    # Use in lambdas, endpoint containers,
                                    # or anywhere you just need to invoke
                                    # endpoints and read/write S3.

pip install 'workbench[aws]'        # + sagemaker SDK + aiobotocore + redis +
                                    #   cryptography. Needed for the orchestration
                                    #   side: building pipelines, deploying
                                    #   endpoints, talking to SageMaker training.

pip install 'workbench[modeling]'   # + xgboost, umap-learn, mordred,
                                    #   cleanlab, ipython. Training-time ML
                                    #   libs (SageMaker training containers
                                    #   have most of these pre-installed).

pip install 'workbench[ui]'         # + plotly, dash, dash-ag-grid,
                                    #   matplotlib. The Workbench Dashboard.

pip install 'workbench[dev]'        # + pytest, pytest-xdist, coverage,
                                    #   flake8, black. Local development.

pip install 'workbench[all]'        # All of the above — typical full install
                                    #   for interactive use, dashboards, and
                                    #   building/deploying pipelines.
```

*Note: shells may interpret square brackets as globs, so the quotes are needed.*

Model-script code running inside SageMaker endpoint containers should
import exclusively from `workbench.endpoints.*` — that's the contract the
endpoint-import-smoke CI job enforces. See `workbench/endpoints/__init__.py`
for the full surface.

### Contributions
If you'd like to contribute to the ADMET Workbench project, you're more than welcome. All contributions will fall under the existing project [license](https://github.com/SuperCowPowers/workbench/blob/main/LICENSE). If you are interested in contributing or have questions please feel free to contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com).

<img align="right" src="docs/images/scp.png" width="180">

® Amazon Web Services, AWS, the Powered by AWS logo, are trademarks of Amazon.com, Inc. or its affiliates
