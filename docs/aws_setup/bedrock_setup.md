# AWS Bedrock Setup

The Workbench ML agent (Bosco) runs Claude through **Amazon Bedrock**, so
prompts are authenticated by your existing Workbench IAM roles, billed through
your AWS account, and never leave AWS.


For what the agent sends to the model and why that boundary is safe, see
[AWS Bedrock Security](bedrock_security.md).

## Enable Bosco

Set `ENABLE_BOSCO` in the personal Workbench **config** for each account. It's also handy to set the dashboard URL so Bosco can open Dashboard
pages for you.

```json
{
    ...
   "DASHBOARD_URL": "<your dashboard URL>",
   "ENABLE_BOSCO": true,
   "WORKBENCH_ROLE": "Workbench-BuilderRole",
    ...
}
```

The config file lives at:

| OS | Path |
| --- | --- |
| macOS / Linux | `~/.workbench/workbench_config.json` |
| Windows | `%LOCALAPPDATA%\Workbench\workbench_config.json` |


!!! note "Which model?"
    The agent defaults to **Claude Opus 5**
    (`us.anthropic.claude-opus-5`). Any Claude model works. Non-Anthropic
    models (Llama, Mistral, Titan) are not supported.

## Verify

```bash
bedrock_verify
```

Uses the same credentials and region as the Workbench REPL, and does a small
round-trip against Claude:

```
Region: us-west-2
Model:  us.anthropic.claude-opus-5
Success: ready
```

To check a different model:

```bash
bedrock_verify us.anthropic.claude-opus-4-8
```

## Cost

Per-token against your AWS account, on the same bill as SageMaker. See
[AWS Service Limits](../admin/aws_service_limits.md) for quota monitoring.

!!! info "Model availability can lag"
    A model listed in the Bedrock catalog is not always callable right away.
    AWS enables models per account, and a newly released one can return an
    access or Marketplace error for a while before it starts working — often
    without any action on your part. Some models also need to be enabled at the
    AWS Organization level, which is outside your account's control.

    If `bedrock_verify` fails on a model you expect to have, wait a few minutes
    and run it again. If it keeps failing, contact Workbench support and we'll
    help sort out where the model is gated.

