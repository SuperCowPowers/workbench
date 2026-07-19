# Amazon Bedrock Setup

Workbench's ML agent runs Claude through **Amazon Bedrock** rather than calling
Anthropic's API directly. This keeps prompts — which contain compound
structures, column names, and model metadata — inside your AWS account's trust
boundary.

!!! tip "Why Bedrock instead of an Anthropic API key?"
    An API key would work and is simpler to set up, but every prompt would
    leave your AWS account and travel to Anthropic's infrastructure. Anthropic
    does not train on API data and retains it only briefly, but for BYOC
    deployments the *contract* is usually the binding constraint, not the
    vendor's retention policy. With Bedrock the request is authenticated by
    your existing Workbench IAM roles, billed through your AWS account, and
    never leaves AWS.

The Workbench Core Stack already grants the Bedrock permissions, so this is a
**one-time console task**: enabling model access in your account.

## Enable model access

Bedrock ships with foundation models **disabled by default**. Until you opt in,
every call returns an access-denied error — which looks exactly like an IAM
misconfiguration and will waste an afternoon.

1. In the AWS Console, search for **Bedrock** and open it. Make sure you're in
   the region your Workbench stacks use.
2. Go to **Model catalog** in the left nav.
3. Pick a Claude model and request access.
4. Some models show a **Submit Use Case Details** form first — fill it in and
   submit.

!!! note "Which model should I enable?"
    Enable at minimum **Claude Opus 4.8** (`anthropic.claude-opus-4-8`) — that's
    what the agent defaults to. Enable additional Claude models if your users
    want to switch between them; the agent will pick up whatever is available.

After submitting, the console drops you back on the model's detail page with no
confirmation banner. That's expected — **check the Model access page** to
confirm the model shows as granted. Access is usually immediate, so don't
resubmit the form if the page looks unchanged.

## Verify

```bash
pip install -U "anthropic[bedrock]"
```

```python
from anthropic import AnthropicBedrockMantle

client = AnthropicBedrockMantle(aws_region="us-west-2")   # your region
msg = client.messages.create(
    model="anthropic.claude-opus-4-8",
    max_tokens=64,
    messages=[{"role": "user", "content": "Reply with the word: ready"}],
)
print(msg.content[0].text)
```

Run this under the same AWS profile the Workbench REPL uses — the client
resolves credentials through the standard AWS chain, so there's nothing extra
to configure.

!!! warning "Reading a 403"
    A `403` means the request reached Bedrock, so your credentials are fine —
    it's authorization. Check, in order:

    1. Model access granted in **this** region (above).
    2. Your Core Stack is current — the Bedrock permissions were added to
       `Workbench-ExecutionRole` and `Workbench-ReadOnlyRole`, so a stack that
       predates them needs a redeploy.
    3. You're running as the role you think you are (`aws sts get-caller-identity`).

    If all three look right, confirm the required action names against the
    current [Bedrock IAM reference](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonbedrock.html).

## Private networking (optional)

By default, Bedrock calls from Workbench compute traverse the public AWS
network path (encrypted, but routed outside your VPC). For customers who
require that traffic stay on private links, add an **interface VPC endpoint**
for Bedrock:

```python
vpc.add_interface_endpoint(
    "BedrockEndpoint",
    service=ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
)
```

Worth doing when a customer asks; not required for the agent to function.

## Cost

Bedrock bills per token against your AWS account — no separate Anthropic
subscription, no minimum. Agent sessions are small compared to training jobs,
but the cost is real and lands on the same bill as SageMaker. See
[AWS Service Limits](../admin/aws_service_limits.md) for where to watch quotas.
