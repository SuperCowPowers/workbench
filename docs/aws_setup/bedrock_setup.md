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
    ...
}
```

The config file lives at:

| OS | Path |
| --- | --- |
| macOS / Linux | `~/.workbench/workbench_config.json` |
| Windows | `%LOCALAPPDATA%\Workbench\workbench_config.json` |


!!! note "Which model?"
    The agent defaults to **Claude Opus 4.8**
    (`us.anthropic.claude-opus-4-8`). Any Claude model works. Non-Anthropic
    models (Llama, Mistral, Titan) are not supported.

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

