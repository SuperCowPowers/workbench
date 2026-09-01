# ADMET Workbench

## Getting Started

```bash
pip install workbench
workbench
```

```
Workbench Version: 0.8.475
Bosco: Claude via Anthropic API key

Welcome to Workbench!

🐶  New to Workbench? Ask me to walk you through building your first model.

Workbench:Bosco>
```

**No AWS account needed to start.** Local mode gives you the whole chain —
`DataSource → FeatureSet → Model → Endpoint` — on your filesystem, training with the
same generated script SageMaker runs. `pub_data` reads public ADMET datasets straight
from S3, so there's real data to model from the first prompt:

```
Workbench:Bosco> build me a solubility model from the aqsol public data
```

Ask Bosco in plain language, or type Python — both land in the same session, and the
variables persist for both of you. See [Local Mode](https://supercowpowers.github.io/workbench/local/).

### Giving Bosco a model

Bosco is the [resident ML agent](https://supercowpowers.github.io/workbench/blogs/ml_agent/).
He needs a path to Claude, and `status` always names the one he took:

- **With an AWS account** — he runs on Bedrock *inside it*, with Zero Data Retention.
  Nothing to install and nothing leaves your account.
- **Without one** — set `ANTHROPIC_API_KEY` and restart. Bring your own key, or email
  [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) and we'll issue
  you a trial key.

### Connecting your AWS account

- [Initial Setup](https://supercowpowers.github.io/workbench/getting_started/)
- [One-time AWS Onboarding](https://supercowpowers.github.io/workbench/aws_setup/core_stack/)
- [Using Bosco](https://supercowpowers.github.io/workbench/bosco/) &nbsp;·&nbsp; [Security & Admin](https://supercowpowers.github.io/workbench/bosco/security/)

📝 [Bosco: Workbench ML Agent](https://supercowpowers.github.io/workbench/blogs/ml_agent/) — a real eleven-turn session, start to finish.

## Chemprop Models

All the rage for the Open ADMET Challenge. ADMET Workbench supports:

- Single Task Chemprop Models
- Multi Task Chemprop Models
- Chemprop + Descriptors Models (MPNN + Molecular Descriptors)
- Foundation Chemprop Models (CheMeleon Pretrained)

Examples: [Deploying Chemprop Models](examples/models/chemprop.py) &nbsp;·&nbsp;
[Deploying Foundation Chemprop Models](examples/models/chemprop_foundation.py)

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

## The Dashboard

Health monitoring, dynamic updates, and a high-level summary across the whole
pipeline, with drill-down views for Incoming Data, Glue Jobs, DataSources,
FeatureSets, Models, and Endpoints. Try it: [Workbench Dashboard Demo](https://workbench-dashboard.com).

## Private SaaS Architecture

*Secure your Data, Empower your ML Pipelines*

Workbench deploys as an AWS Stack inside your own cloud environment (BYOC), so your
data never leaves it — and plugin support lets you tailor it to your own workflows.
See the [Private SaaS Architecture](https://docs.google.com/presentation/d/1f_1gmE4-UAeUDDsoNdzK_d_MxALFXIkxORZwbJBjPq4/edit?usp=sharing) deck.

## Documentation

The ADMET Workbench framework makes AWS® both easier to use and more powerful: a simple
Python API over Glue, Athena, Feature Store, Models, and Endpoints, with web interfaces
on top of it.

[Workbench Docs](https://supercowpowers.github.io/workbench/) covers the Python API
in depth, fully searchable, with code examples throughout. Runnable versions of every
example live in [`examples/`](https://github.com/SuperCowPowers/workbench/blob/main/examples).

Workbench takes something genuinely complex — the full set of AWS ML services — and
makes it less complex, so there's conceptual documentation too:
[Workbench Presentations](https://supercowpowers.github.io/workbench/presentations/).

## AWS Marketplace

Workbench is on the AWS Marketplace as a [Dashboard for ML Pipelines](https://aws.amazon.com/marketplace/pp/prodview-5idedc7uptbqo)
and can be billed through AWS.

## Installation extras

```bash
pip install workbench               # API + REPL + orchestration (the default)
pip install 'workbench[ui]'         # + plotly, dash — the Workbench Dashboard
pip install 'workbench[modeling]'   # + torch, chemprop, ray[tune], optuna
pip install 'workbench[misc]'       # + umap-learn
pip install 'workbench[dev]'        # + pytest, coverage, flake8, black
pip install 'workbench[all]'        # everything above
```

Quotes are needed — shells read square brackets as globs.

**Cleanlab/Datalab** workflows aren't bundled in any extra. Install
`cleanlab[datalab]>=2.8.0` before calling `cleanlab_model()`.

## Questions?

The SuperCowPowers team is happy to answer any questions about AWS and Workbench —
[workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or
[Discord](https://discord.gg/WHAJuz8sw8).

**Beta Program** — if your company would like to be a Workbench Beta Tester, get in
touch at the same address.

**Contributions** are welcome, and fall under the existing project
[license](https://github.com/SuperCowPowers/workbench/blob/main/LICENSE).

<img align="right" src="docs/images/scp.png" width="180">

® Amazon Web Services, AWS, the Powered by AWS logo, are trademarks of Amazon.com, Inc. or its affiliates
