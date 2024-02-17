
# SageWorks Dashboard
This is the Python CDK for deploying the SageWorks Dashboard web service AWS Stack. The AWS Stack has quite a few components.

- ECS Fargate: Deploying a Fargate service for running containerized applications.
- Application Load Balancer: Configured to balance traffic to the Fargate service, with optional public access and HTTPS support.
- VPC: A Virtual Private Cloud with subnets spread across 2 Availability Zones for high availability. The VPC includes:
  - Public Subnets: For resources that need to be accessible from the internet, such as the Load Balancer when in public mode.
  - Private Subnets: For resources that should not be directly accessible from the internet, such as the ECS tasks and the Redis cluster.
- NAT Gateways: Deployed in each public subnet to allow outbound internet access for resources in the private subnets.
- ElastiCache Redis Cluster: A Redis cluster in Amazon ElastiCache, deployed in the private subnets for caching purposes, accessible only from within the VPC.
- Security Groups: Configured to control access to the ECS tasks and the Redis cluster, with rules based on IP whitelisting and AWS managed prefix lists.


## Setting SAGEWORKS_CONFIG before Synth/Diff/Deploy
```
export SAGEWORKS_CONFIG=/full/path/to/deploy_config.json
```
At this point you can now synthesize the CloudFormation template for this code.

## Synth, Diff, and Deploy
```
$ cdk synth
$ cdk diff
$ cdk deploy
```


## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

Enjoy!
