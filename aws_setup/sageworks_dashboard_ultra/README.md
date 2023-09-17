
# SageWorks Dashboard
This is the Python CDK for deploying the SageWorks Dashboard web service AWS Stack. The AWS Stack has quite a few components.




## Setting ENV before Synth/Deploy
```
SAGEWORKS_BUCKET=my-sageworks-bucket
SAGEWORKS_WHITELIST=10.16.0.0/16,1.2.3.4/32
SAGEWORKS_SUBNET_IDS=subnet-123456
SAGEWORKS_VPC_ID=vpc-123987
```
At this point you can now synthesize the CloudFormation template for this code.

## Synth and Deploy
```
$ cdk synth
$ cdk deploy
```


## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

Enjoy!
