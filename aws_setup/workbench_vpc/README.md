# Workbench VPC Stack

Creates a small, cost-optimized VPC for Workbench **Batch compute**: one VPC
across two Availability Zones, each with a public and a private subnet, an
Internet Gateway, and a single NAT gateway. The CDK `ec2.Vpc` construct wires the
subnets, route tables, IGW, and NAT automatically.

Deploy this **only if you don't already have a VPC** for the
[Compute stack](../../docs/aws_setup/compute_stack.md). Workbench is
bring-your-own-VPC — this stack is a convenience for fresh accounts.

```bash
cd workbench/aws_setup/workbench_vpc
pip install -r requirements.txt
cdk bootstrap   # if this account/region isn't bootstrapped yet
cdk deploy
```

After deploy, copy the stack outputs into your Workbench config and then deploy
`workbench_compute`:

- `VpcId` → `WORKBENCH_VPC_ID` (string)
- `PrivateSubnetIds` → `WORKBENCH_SUBNET_IDS` (JSON array of the two private subnet ids)

Optional config overrides (defaults shown): `WORKBENCH_VPC_CIDR` (`10.0.0.0/16`),
`WORKBENCH_VPC_MAX_AZS` (`2`), `WORKBENCH_VPC_NAT_GATEWAYS` (`1`).

> The NAT gateway bills ~$32/month while it exists. To pause costs you can
> `cdk destroy` this stack (after removing the compute stack that uses it).
