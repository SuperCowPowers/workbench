# Deploy the Workbench Dashboard Stack

<figure style="float: right; width: 300px;">
<img alt="workbench_new_light" src="https://github.com/SuperCowPowers/workbench/assets/4806709/5f8b32a2-ed72-45f2-bd96-91b7bbbccff4" style="padding-left: 10px; padding-top: -20px; ">
</figure>

Deploying the Dashboard Stack is reasonably straight forward, it's the same approach as the [Core Stack](core_stack.md) that you've already deployed.

Please review the [Stack Details](#stack-details) section to understand all the AWS components that are included and utilized in the Workbench Dashboard Stack.

## Prerequisites
The Dashboard serves over HTTPS, so a domain and SSL certificate need to be in place first. See [Domain and SSL Certificate Setup](domain_cert_setup.md) and confirm both before you deploy:

- Your domain is registered in Route 53 for **this** AWS account
- Your ACM certificate is in the **Issued** state
- `WORKBENCH_CERTIFICATE_ARN` is set in your Workbench config file

```json
"WORKBENCH_ROLE": "Workbench-ExecutionRole",
"WORKBENCH_CERTIFICATE_ARN": "arn:aws:acm:<region>:<account>:certificate/123-987-456-123-456789012",
```

## Deploying the Dashboard Stack

!!! note "AWS Stuff"
    Activate your AWS Account that's used for Workbench deployment. For this one time install you should use an Admin Account (or an account that had permissions to create/update AWS Stacks)

  ```bash
  cd workbench/aws_setup/workbench_dashboard
  export WORKBENCH_CONFIG=/full/path/to/config.json
  pip install -r requirements.txt
  cdk bootstrap
  cdk deploy
  ```

## Point Your Domain at the Load Balancer
The stack creates a new load balancer, so the last step is a Route 53 **A** record that routes your domain to it.

- Go to the [Route 53 Console](https://console.aws.amazon.com/route53/)
- Click **Hosted zones** (left panel) and select your domain
- Click **Create record** (or click an existing **A** record to update it)
- Leave **subdomain** blank for the top-level record
- Click the **Alias** toggle (important)
- For **Route traffic to**:
    - *Alias to Application and Classic Load Balancer*
    - Select the AWS Region
    - Use the chooser box to find your load balancer

Repeat for any additional subdomain on your certificate (e.g. `www.your-domain.com`).

!!! tip "Finding the Load Balancer"
    The chooser box usually finds it for you. If you need to look it up directly, load balancers live under the [EC2 Console](https://console.aws.amazon.com/ec2/) → **Load Balancing** → **Load Balancers**. The **DNS name** field is what you want, something like `dualstack.workbe-workb-xyzabc-123456.us-west-2.elb.amazonaws.com`.

## Stack Details
!!! question inline end "AWS Questions?"
    There's quite a bit to unpack when deploying an AWS powered Web Service. We're happy to help walk you through the details and options. Contact us anytime for a free consultation.
    
- ECS Fargate
- Load Balancer
- 2 Availability Zones
  - VPCs / Nat Gateways
- ElasticCache Cluster (shared Redis Caching)

### AWS Stack Benefits

1. **Scalability**: Includes an Application Load Balancer and uses ECS with Fargate, and ElasticCache for more robust scaling options.
1. **Higher Security**: Utilizes security groups for both the ECS tasks, load balancer, plus VPC private subnets for Redis and the utilization of NAT Gateways.

!!! warning "AWS Costs"
    Deploying the Workbench Dashboard does incur some monthly AWS costs.

### Public vs Private Deployment
The same stack deploys either an internal (private) or an internet-facing (public) dashboard, controlled by config:

- `WORKBENCH_DASHBOARD_PUBLIC` (default `false`): set to `true` for an internet-facing load balancer with open 443/80 ingress. Leave unset/`false` for an internal load balancer reachable only via your whitelisted IPs and prefix lists.
- `WORKBENCH_DASHBOARD_TASK_COUNT` (default `1`): number of Fargate tasks to run behind the load balancer.
