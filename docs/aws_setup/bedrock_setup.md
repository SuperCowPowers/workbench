# AWS Bedrock Setup

The Workbench ML agent runs Claude through **Amazon Bedrock**, so prompts are
authenticated by your existing Workbench IAM roles, billed through your AWS
account, and never leave AWS.

The Core Stack already grants the Bedrock permissions. Setup is a one-time
console visit.

For what the agent sends to the model and why that boundary is safe, see
[AWS Bedrock Security](bedrock_security.md).

## Setup

Once per AWS account, as an **administrator**:

- Open **Bedrock** in the console, in your Workbench region.
- **Model catalog** → select a Claude model.
- Submit the **Use Case Details** form if it appears.
- Open the model in the **playground** and send it a prompt.

The playground prompt is what enables the model account-wide. There's no
confirmation screen — verify below.

!!! note "Which model?"
    The agent defaults to **Claude Opus 4.8**
    (`us.anthropic.claude-opus-4-8`). Any Claude model works — repeat the
    playground step for each one you want. Non-Anthropic models (Llama,
    Mistral, Titan) are not supported.

## Verify

```bash
bedrock_verify
```

Uses the same credentials and region as the Workbench REPL, and does a small
round-trip against Claude:

```
Region: us-west-2
Model:  us.anthropic.claude-opus-4-8
Success: ready
```

To check a different model:

```bash
bedrock_verify us.anthropic.claude-sonnet-5
```

## Cost

Per-token against your AWS account, on the same bill as SageMaker. See
[AWS Service Limits](../admin/aws_service_limits.md) for quota monitoring.

## Troubleshooting

### 403 / AccessDenied

Credentials are fine — this is authorization:

1. **`aws-marketplace:Subscribe`** — nobody ran the playground step for this
   model. See Setup.
2. **Service Control Policy** — an org-level deny overrides the account. Needs
   org access to allow `bedrock:*`.
3. **Stale Core Stack** — redeploy if it predates the Bedrock permissions.
4. **Wrong role** — check `aws sts get-caller-identity`.

### 404 / model does not exist

Model availability varies by region, and Claude 4.x needs the `us.` inference
profile prefix — not a bare model id.
