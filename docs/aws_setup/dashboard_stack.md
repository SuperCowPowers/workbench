# Deploy the Workbench Dashboard Stack

<figure style="float: right; width: 300px;">
<img alt="workbench_new_light" src="https://github.com/SuperCowPowers/workbench/assets/4806709/5f8b32a2-ed72-45f2-bd96-91b7bbbccff4" style="padding-left: 10px; padding-top: -20px; ">
</figure>

Deploying the Dashboard Stack is reasonably straight forward, it's the same approach as the [Core Stack](core_stack.md) that you've already deployed.

Please review the [Stack Details](#stack-details) section to understand all the AWS components that are included and utilized in the Workbench Dashboard Stack.

## Prerequisites
The Dashboard serves over HTTPS, so a domain and SSL certificate need to be in place first. See [Domain and SSL Certificate Setup](domain_cert_setup.md) and confirm both before you deploy:

- You've picked the domain name the Dashboard will serve on
- A certificate covering that name is in ACM **in this account and region**, in the **Issued** state
- `WORKBENCH_CERTIFICATE_ARN` is set in your Workbench config file
- For a private dashboard, `WORKBENCH_PREFIX_LISTS` or `WORKBENCH_WHITELIST` names the networks allowed to reach it (see [Who Can Reach the Dashboard](#who-can-reach-the-dashboard))

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
The stack creates a new load balancer, so the last step is a DNS record routing your domain to it. How you add that record depends on which option you took in [Domain and SSL Certificate Setup](domain_cert_setup.md).

First, grab the load balancer's DNS name — you'll need it either way:

```bash
aws cloudformation describe-stacks --stack-name WorkbenchDashboard \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDnsName`].OutputValue' --output text
```

It looks something like `internal-workbe-workb-xyzabc-123456.us-west-2.elb.amazonaws.com`. You can also find it in the [EC2 Console](https://console.aws.amazon.com/ec2/) under **Load Balancing** → **Load Balancers** → **DNS name**.

Now follow **one** of the two sections below — whichever matches the option you took on the [Domain and SSL Certificate Setup](domain_cert_setup.md) page. They're alternatives, not sequential steps.

### Option 1: Company Subdomain (External DNS)
Alias records are a Route 53 feature, so for an external DNS provider you need a **CNAME**. Your company's DNS is almost certainly managed by IT — admin rights on the AWS account don't cover a domain registered elsewhere — so this step is a hand-off. Send them the record:

> Hi — could you please add a DNS record for our ML Dashboard?
>
> - **Type:** CNAME
> - **Name:** `ml-dashboard.yourcompany.com`
> - **Value:** `internal-workbe-workb-xyzabc-123456.us-west-2.elb.amazonaws.com`
> - **TTL:** 300
>
> It points at an internal AWS load balancer, so it resolves to private addresses (`10.x.x.x`) reachable from our VPC and over VPN. No public exposure.

The name you request must be one your certificate covers — either the exact name on the certificate, or any single label under a wildcard like `*.dev.yourcompany.com`.

!!! warning "Private dashboards resolve to private IPs"
    An internal load balancer (`WORKBENCH_DASHBOARD_PUBLIC` unset or `false`) publishes only private VPC addresses. Users reach it from inside the VPC or over VPN. That works fine in a public DNS zone, but some organizations prefer to keep internal names in split-horizon DNS — your DNS team will know which they want.

---

### Option 2: Dedicated Route 53 Domain
Use an **A** record with an alias — it resolves straight to the load balancer with no extra DNS hop.

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

---

### Verify
Once DNS propagates:

```bash
curl -sSf https://your-dashboard-domain.com/health && echo " <- dashboard is up"
```

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

- `WORKBENCH_DASHBOARD_PUBLIC` (default `false`): set to `true` for an internet-facing load balancer with open 443/80 ingress. Leave unset/`false` for an internal load balancer reachable only from the networks you whitelist.
- `WORKBENCH_DASHBOARD_TASK_COUNT` (default `1`): number of Fargate tasks to run behind the load balancer.

### Who Can Reach the Dashboard
A private dashboard's load balancer starts with **no** inbound access. You grant it with one or both of these, whichever suits how your network is managed. Both take a comma separated list.

**`WORKBENCH_PREFIX_LISTS`** — [AWS managed prefix lists](https://docs.aws.amazon.com/vpc/latest/userguide/managed-prefix-lists.html), each a named set of CIDR blocks maintained in one place:

```json
"WORKBENCH_PREFIX_LISTS": "pl-00b18c17bdeb8794c, pl-0421d7c17b5a9c605",
```

Prefer these if your network team already publishes them. The CIDRs live in the prefix list, so when a VPN range changes your team updates the list once and every stack referencing it follows — no redeploy. A prefix list shared from another account via AWS RAM works fine; just reference its ID.

**`WORKBENCH_WHITELIST`** — raw CIDR blocks, when you have no prefix lists to point at:

```json
"WORKBENCH_WHITELIST": "10.49.0.0/16, 203.0.113.42/32",
```

CIDR notation is required — a bare IP address is rejected, so use `/32` for a single host.

Both apply to port 443 on the load balancer and port 6379 on Redis. To find the range your users actually arrive from, check the address a VPN-connected machine holds (`ifconfig` on macOS/Linux, `ipconfig` on Windows) and ask your network team for the pool it belongs to.

!!! warning "Private with an empty whitelist is unreachable"
    With no whitelist and no prefix lists, the load balancer gets no rule on port 443 and nothing can connect. Tasks pass their health checks, the stack reports success, and browsers simply time out — there's no error to find. `cdk deploy` refuses to synth this combination, but it's worth knowing the shape of the failure.
