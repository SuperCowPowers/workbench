# WorkbenchCompute AWS Stack

The **WorkbenchCompute** stack provisions the AWS Batch infrastructure that runs
Workbench ML pipelines as managed, parallel jobs. It is deployed *after* the
[WorkbenchCore stack](core_stack.md).

!!! note "Do you actually need this stack?"
    WorkbenchCompute is only required to run pipelines on **AWS Batch** (fan-out
    of many model/feature pipelines). You do **not** need it to:

    - run pipeline scripts **locally** (`python my_pipeline.py`), or
    - train and deploy **SageMaker models and endpoints** (that is the Core stack + SageMaker).

    If you are just getting started, you can skip this stack and add it later
    when you want Batch parallelism.

## What the stack creates

- **Fargate Compute Environment** (`workbench-compute-env`) — serverless Batch
  compute, capped at 16 vCPU to avoid AWS throttling.
- **Job Queue** (`workbench-job-queue`).
- **Job Definitions** in three tiers — `workbench-batch-small` (2 vCPU / 4 GB),
  `-medium` (4 vCPU / 8 GB), `-large` (8 vCPU / 16 GB) — each running the
  `py312-ml-pipelines` ECR image.
- **SQS FIFO queue + DLQ + a trigger Lambda** that reads pipeline scripts from
  S3 and submits Batch jobs with the correct `dependsOn` ordering (this is what
  turns a `pipelines.json` manifest into an executed DAG).
- **Failure notifications** — an SNS topic, a failure-handler Lambda, an
  EventBridge rule on failed jobs, and a CloudWatch alarm on the DLQ.

## Why a VPC is required

Batch Fargate jobs must pull the pipeline container from ECR, which needs a
network egress path. The **default VPC cannot do this** for Batch, so the stack
intentionally refuses to deploy without an explicit VPC:

```
ValueError: Please provide the Workbench Config entry: "WORKBENCH_VPC_ID":"vpc-...".
Default VPC networks cannot pull ECR images for Batch jobs.
```

Workbench is **bring-your-own-VPC** — it looks up an existing VPC rather than
creating one. If you already have a suitable VPC, skip to
[Wire the VPC into your config](#wire-the-vpc-into-your-workbench-config). If not,
deploy the small `workbench_vpc` stack below.

## Create a VPC (if you don't have one)

The [`workbench_vpc`](https://github.com/SuperCowPowers/workbench/tree/main/aws_setup/workbench_vpc)
stack builds **one VPC** across two Availability Zones — each with a public and a
private subnet — plus an Internet Gateway and a single NAT gateway. Batch runs in
the **private** subnets and reaches ECR through the NAT.

```
            VPC (10.0.0.0/16)  — spans the region
   ┌──────────────── AZ a ────────────┬──────────── AZ b ────────────┐
   │  public subnet  ──► IGW          │  public subnet  ──► IGW       │
   │     └─ NAT GW                     │                              │
   │  private subnet ──► NAT          │  private subnet ──► NAT       │
   └──────────────────────────────────┴──────────────────────────────┘
        WORKBENCH_SUBNET_IDS = [private-a, private-b]   ← Batch runs here
```

!!! tip "Subnets are free — the NAT gateway is the cost"
    A VPC, its subnets, the Internet Gateway, and route tables cost nothing. The
    only recurring charge here is the **NAT gateway (~$32/month + data)**. A
    single NAT serves **both** private subnets, so two-AZ resilience costs you
    nothing extra — there is no reason to drop to a single AZ.

Deploy it like any other Workbench stack:

```bash
cd workbench/aws_setup/workbench_vpc
pip install -r requirements.txt
cdk bootstrap   # only if this account/region isn't bootstrapped yet
cdk deploy
```

When it finishes, CDK prints two stack outputs — note them for the next step:

```
Outputs:
WorkbenchVpc.VpcId = vpc-0123456789abcdef0
WorkbenchVpc.PrivateSubnetIds = subnet-0aaa...,subnet-0bbb...
```

!!! tip "Optional overrides"
    Defaults are CIDR `10.0.0.0/16`, 2 AZs, 1 NAT gateway. Override via config
    (`WORKBENCH_VPC_CIDR`, `WORKBENCH_VPC_MAX_AZS`, `WORKBENCH_VPC_NAT_GATEWAYS`)
    before `cdk deploy` if needed.

## Wire the VPC into your Workbench config

Add the VPC id and the **private** subnet ids to your site config JSON (the file
referenced by `WORKBENCH_CONFIG`, e.g. `~/.workbench/<name>.json`).
`WORKBENCH_SUBNET_IDS` must be a **JSON array**:

```json
{
  "WORKBENCH_VPC_ID": "vpc-0123456789abcdef0",
  "WORKBENCH_SUBNET_IDS": ["subnet-0aaa...", "subnet-0bbb..."]
}
```

(`WORKBENCH_VPC_ID` = the `VpcId` output; `WORKBENCH_SUBNET_IDS` = the
`PrivateSubnetIds` output, split into a JSON array.)

## Deploy the Compute Stack

```bash
cd workbench/aws_setup/workbench_compute
pip install -r requirements.txt
cdk deploy
```

`cdk diff` should now show the Batch resources instead of the VPC error, and the
printed configuration should list your `WORKBENCH_VPC_ID` and
`WORKBENCH_SUBNET_IDS`.

!!! success "Verify"
    Confirm the launcher CLI is available:
    ```bash
    ml_pipeline_launcher --help
    ```
    Then, from a directory containing a `pipelines.json` (or a standalone
    pipeline script), launch a run and watch the job reach `RUNNING` in the AWS
    Batch console:
    ```bash
    ml_pipeline_launcher --dry-run --all   # preview what would launch
    ml_pipeline_launcher --all             # submit them
    ```

## Cost & teardown

The NAT gateway in the `workbench_vpc` stack bills ~**$32/month** plus data
whether or not jobs are running. To pause costs, tear down the stacks that use it
(`cdk destroy` in `workbench_compute`, then `workbench_vpc`) and redeploy when you
need Batch again — CDK cleans up the NAT, EIP, and subnets for you.

---

If you run into any issues, the SuperCowPowers team is happy to help with AWS for
**free** — reach us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
or on [Discord](https://discord.gg/WHAJuz8sw8).
