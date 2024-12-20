# Getting Started
For the initial setup of Workbench we'll be using the Workbench REPL. When you start `workbench` it will recognize that it needs to complete the initial configuration and will guide you through that process.

!!!tip inline end "Need Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and Workbench. So please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

## Initial Setup/Config
**Notes:** Workbench uses your existing AWS account/profile/SSO. So if you don't already have an AWS Profile or SSO Setup you'll need to do that first [AWS Setup](../aws_setup/aws_setup.md)

Okay so **after** you've completed your [AWS Setup](../aws_setup/aws_setup.md) you can now install Workbench.

```
> pip install workbench
> workbench <-- This starts the REPL

Welcome to Workbench!
Looks like this is your first time using Workbench...
Let's get you set up...
AWS_PROFILE: my_aws_profile
WORKBENCH_BUCKET: my-company-workbench
[optional] REDIS_HOST(localhost): my-redis.cache.amazon (or leave blank)
[optional] REDIS_PORT(6379):
[optional] REDIS_PASSWORD():
[optional] WORKBENCH_API_KEY(open_source): my_api_key (or leave blank)
```
**That's It:** You're now all set. This configuration only needs to be **ONCE** :)

### Data Scientists/Engineers
- Workbench REPL: [Workbench REPL](../repl/index.md)
- Using Workbench for ML Pipelines: [Workbench API Classes](../api_classes/overview.md)
- SCP Workbench Github: [Github Repo](https://github.com/SuperCowPowers/workbench)


### AWS Administrators
For companies that are setting up Workbench on an internal AWS Account: [Company AWS Setup](../aws_setup/core_stack.md)

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

- Workbench Core Classes: [Core Classes](../core_classes/overview.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
