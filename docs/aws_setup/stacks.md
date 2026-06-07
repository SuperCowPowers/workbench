# The Workbench Stacks

Workbench deploys to AWS as a set of **CDK stacks**. There are three, and only
the **Core** stack is required — the others are opt-in depending on what you need.

| Stack | Provisions | Required? | Needs a VPC? |
|---|---|---|---|
| **Core** | `Workbench-ExecutionRole` + AWS Glue role; the IAM/identity foundation everything else assumes | **Yes** — deploy first | No |
| **Compute** | AWS Batch (Fargate) compute env, job queue, job definitions, an SQS + Lambda DAG trigger, and failure alerts | Optional | Yes |
| **Dashboard** | ECS Fargate web app + Application Load Balancer + Redis (ElastiCache); `full` or `lite` | Optional | Builds its own |

## Core Stack — *required*

The foundation. It creates the `Workbench-ExecutionRole` (the least-privilege
role that all Workbench access flows through) and an associated AWS Glue role.
Deploy this first; everything else references it.

→ [Core Stack setup](core_stack.md)

### Access & roles

Workbench centralizes AWS access through the `Workbench-ExecutionRole`: users (or
their SSO group) are granted permission to *assume* that role, and the role
itself holds the least-privilege policies for what Workbench can do. This keeps
"what Workbench is allowed to touch" in one place. For a conceptual walkthrough,
see the [Workbench AWS Access Management slide deck](https://docs.google.com/presentation/d/1_KwbaBsyBoiWW_8SEallHg8RMsi9FdK10dr2wwzo3CA/edit?usp=sharing).

Granting access is a one-time admin step after the Core stack is deployed:

- [Grant access — SSO Users](sso_assume_role.md)
- [Grant access — IAM Users](iam_assume_role.md)

## Compute Stack — *optional*

Adds **AWS Batch** so Workbench pipelines can run as managed, parallel jobs
(fan-out across many models/feature sets) instead of locally. It requires a VPC
with egress to ECR — a small `workbench_vpc` helper stack can create a
cost-optimized one if you don't already have a VPC.

You do **not** need this stack to run pipelines locally or to train/deploy
SageMaker models and endpoints — only for Batch fan-out.

→ [Compute Stack setup](compute_stack.md)

## Dashboard Stack — *optional*

Deploys the Workbench **Dashboard** as a team web app (ECS Fargate behind a load
balancer, with shared Redis caching) so everyone can browse the ML artifacts
Workbench creates. It builds its own VPC and networking.

Comes in two flavors:

- **`full`** — load balancer, multi-AZ, ElastiCache — robust for production use.
- **`lite`** — a trimmed, lower-cost deployment for tight budgets.

→ [Dashboard Stack setup](dashboard_stack.md) · optionally add a
[custom domain + SSL cert](domain_cert_setup.md).

## Deploy order

1. **Core** stack (required) → then grant role access (SSO/IAM).
2. *(optional)* **Compute** stack — if you want Batch pipeline fan-out.
3. *(optional)* **Dashboard** stack — if you want the team web UI.

!!! tip "Need a hand?"
    The SuperCowPowers team helps new users with AWS for **free** —
    [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on
    [Discord](https://discord.gg/WHAJuz8sw8).
