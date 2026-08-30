# Getting Started

!!!tip inline end "Need Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and ADMET Workbench. So please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)

## Install and run

```bash
pip install workbench
workbench
```

`workbench` starts the REPL. With no AWS configuration it comes up in **local
mode**: artifacts live on your filesystem, so there's no account to create, no
credentials to manage, and nothing to pay for.

The base install carries XGBoost, RDKit, and the descriptor stack — everything on
this page. Two extras add the rest:

| Extra | Adds |
| --- | --- |
| `workbench[modeling]` | Chemprop and PyTorch models (pulls torch, a large download) |
| `workbench[ui]` | The dashboard and its plugin pages |

## Build a model

`pub_data` reads public datasets from S3 anonymously, and the local classes are
already bound at the prompt, so this runs as-is:

```python
df = pub_data.get("comp_chem/aqsol/aqsol_public_data")

ds = DataSource(df, name="aqsol_local")
fs = ds.to_features("aqsol_local_features", id_column="ID")
model = fs.to_model(
    "aqsol-local-reg",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.XGBOOST,
    target_column="solubility",
    feature_list=["molwt", "mollogp", "tpsa", "numrotatablebonds"],
)

model.get_inference_metrics()
```

Training runs the same model script SageMaker runs, so a model that works here
publishes and produces the same model. See [Local Mode](../local/index.md) for
scoring, endpoints, and the rest of the local API.

## Or just ask

[Bosco](../bosco/index.md), the Workbench ML agent, runs in local mode too. Give it
an Anthropic API key and it writes and runs that chain for you:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
workbench
```

`Bosco` appears in the prompt, and anything you type that isn't Python goes to it:

```
Workbench:Bosco> build me a solubility model from the aqsol public data
Workbench:Bosco> show me a predicted vs actual plot
Workbench:Bosco> what are the nearest neighbors of the worst outlier?
```

It works in your live session, so the variables it creates stay in your namespace.
Prompts go to the Anthropic API rather than an AWS account in this mode — `status`
always names where they go. See [Security & Admin](../bosco/security.md).

## Connect your AWS account

Publishing a model, deployed endpoints, monitoring, and the dashboard all need an
AWS account. ADMET Workbench uses your existing AWS account/profile/SSO — if you
don't have one yet, start with [AWS Setup](../aws_setup/aws_setup.md).

Then run `aws_setup()` from the REPL:

```
aws_setup()

AWS_PROFILE: my_aws_profile
WORKBENCH_BUCKET: my-company-workbench
[optional] REDIS_HOST(localhost): my-redis.cache.amazon (or leave blank)
[optional] REDIS_PORT(6379):
[optional] REDIS_PASSWORD():
[optional] DASHBOARD_URL():
[optional] ENABLE_BOSCO -- run the Bosco ML agent? (y/N):
```

It writes the config, prints the one environment variable to export, and exits.
Add that line, open a new terminal, and run `workbench` again — you only do this
**once**.

### Data Scientists/Engineers
- Workbench REPL: [Workbench REPL](../repl/index.md)
- Using Workbench for ML Pipelines: [Workbench API Classes](../api_classes/overview.md)
- SCP Workbench Github: [Github Repo](https://github.com/SuperCowPowers/workbench)


### AWS Administrators
For companies that are setting up ADMET Workbench on an internal AWS Account: [Company AWS Setup](../aws_setup/core_stack.md)

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

- Workbench Core Classes: [Core Classes](../core_classes/overview.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
