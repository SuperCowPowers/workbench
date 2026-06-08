# Accessing Redis (SSM Tunnel)

The Workbench **Redis (ElastiCache) cluster lives in private subnets** and is not
reachable from a laptop directly. In-VPC consumers (the Dashboard, AWS Batch
jobs) reach it automatically; to connect from your machine — e.g. to inspect or
clear cache keys — open a short-lived **SSM port-forwarding tunnel**.

!!! tip "Why SSM instead of a VPN or SSH bastion?"
    SSM Session Manager port-forwarding needs **no SSH key, no public IP, and no
    inbound security-group rule** — the tunnel instance sits in a *private*
    subnet and talks outbound only. It authenticates with your existing AWS SSO
    credentials, so there's no VPN client, certificate setup, or `.pem` to manage
    (and nothing to leak). A Client VPN only earns its complexity when a whole
    team needs routine access to many private resources.

## Prerequisite

Install the AWS CLI Session Manager plugin once:

```bash
brew install session-manager-plugin    # macOS
```
(Other platforms: see the [AWS docs](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html).)

## Open a tunnel

Run the helper with admin AWS credentials (the same profile you use for `cdk deploy`):

```bash
export WORKBENCH_CONFIG=/path/to/<profile>_admin.json
python workbench/scripts/admin/redis_tunnel.py
```

It discovers the VPC / private subnet from your config and the Redis endpoint from
ElastiCache, brings up (or starts) a tiny keyless instance, waits for it to
register with SSM, and **opens the tunnel in the foreground**. Connect through the
forwarded local port from another terminal:

```bash
redis-cli -h localhost -p 6380
```

When you're done, **press Ctrl-C** in the tunnel terminal — it closes the tunnel
and **stops the instance automatically** (no standing cost). Run the script again
to reconnect (it restarts the same instance).

## Tear down

Ctrl-C already stops the instance on exit, so nothing runs between sessions (a
stopped `t3.micro` costs only its small EBS volume, and reconnecting just restarts
it). To remove it entirely — instance **and** security group — when you no longer
need tunnels:

```bash
python workbench/scripts/admin/redis_tunnel.py --terminate
```

## How it works

1. Reads `WORKBENCH_VPC_ID` / `WORKBENCH_SUBNET_IDS` from the active config and
   looks up the Redis endpoint from ElastiCache (nothing hardcoded).
2. Ensures a minimal IAM instance profile (`AmazonSSMManagedInstanceCore`) and an
   egress-only security group exist.
3. Launches a keyless Amazon Linux 2023 instance in a **private** subnet
   (outbound reaches SSM via the VPC's NAT; no public IP, IMDSv2-only).
4. SSM forwards `localhost:<port>` → through the instance → to the private Redis
   endpoint. The instance is the conduit; nothing is exposed to the internet.

!!! tip "Need a hand?"
    The SuperCowPowers team helps new users with AWS for **free** —
    [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on
    [Discord](https://discord.gg/WHAJuz8sw8).