# Getting Started
For the initial setup of SageWorks we'll be using the SageWorks REPL. When you start `sageworks` it will recognize that it needs to complete the initial configuration and will guide you through that process.

!!!tip inline end "Need Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and SageWorks. So please contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

## Initial Setup/Config
**Notes:** SageWorks uses your existing AWS account/profile/SSO. So if you don't already have an AWS Profile or SSO Setup you'll need to do that first [AWS Setup](../aws_setup/aws_setup.md)

Okay so **after** you've completed your [AWS Setup](../aws_setup/aws_setup.md) you can now install SageWorks.

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

### Data Scientists/Engineers
- SageWorks REPL: [SageWorks REPL](../repl/index.md)
- Using SageWorks for ML Pipelines: [SageWorks API Classes](../api_classes/overview.md)
- SCP SageWorks Github: [Github Repo](https://github.com/SuperCowPowers/sageworks)


### AWS Administrators
For companies that are setting up SageWorks on an internal AWS Account: [Company AWS Setup](../aws_setup/core_stack.md)

## Additional Resources

<img align="right" src="../images/scp.png" width="180">

- SageWorks Core Classes: [Core Classes](../core_classes/overview.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
