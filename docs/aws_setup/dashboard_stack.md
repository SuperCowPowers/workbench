# Deploy the SageWorks Dashboard Stack

<figure style="float: right; width: 300px;">
<img alt="sageworks_new_light" src="https://github.com/SuperCowPowers/sageworks/assets/4806709/5f8b32a2-ed72-45f2-bd96-91b7bbbccff4" style="padding-left: 10px; padding-top: -20px; ">
</figure>

Deploying the Dashboard Stack is reasonably straight forward, it's the same approach as the [Core Stack](core_stack.md) that you've already deployed.

Please review the [Stack Details](#stack-details) section to understand all the AWS components that are included and utilized in the SageWorks Dashboard Stack.

## Deploying the Dashboard Stack

!!! note "AWS Stuff"
    Activate your AWS Account that's used for SageWorks deployment. For this one time install you should use an Admin Account (or an account that had permissions to create/update AWS Stacks)

  ```bash
  cd sageworks/aws_setup/sageworks_dashboard_full
  export AWS_PROFLE=<aws_admin_account>
  export SAGEWORKS_BUCKET=<name of your S3 bucket>
  (optional) export SAGEWORKS_SSO_GROUP=<your SSO group>
  pip install -r requirements.txt
  cdk bootstrap
  cdk deploy
  ```

## Stack Details
!!! question inline end "AWS Questions?"
    There's quite a bit to unpack when deploying an AWS powered Web Service. We're happy to help walk you through the details and options. Contact us anytime for a free consultation.
    
- ECS Fargate
- Load Balancer
- 2 Availability Zones
  - 4 VPC (2 public and 2 private)
- 2 Nat Gateways
- ElasticCache Cluster (shared Redis Caching)

### Pros

1. **Scalability**: Includes an Application Load Balancer and uses ECS with Fargate, and ElasticCache for more robust scaling options.
1. **Higher Security**: Utilizes security groups for both the ECS tasks, load balancer, plus VPC private subnets for Redis and the utilization of NAT Gateways.

!!! warning inline end "AWS Costs"
    Deploying the SageWorks Dashboard does incur some monthly AWS costs. If you'd prefer a simpler, less costly, deployment please contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)

### Cons

1. **Cost**: This setup uses additional AWS services like Elasticache, Load Balancer, and NAT Gateways.
