# The Workbench ML Agent
!!! tip inline end "Already Using Workbench?"
    Bosco ships with the [Workbench REPL](../repl/index.md) — see
    [Using Bosco](../bosco/index.md) for how to drive him. Model access is a
    one-time setup: [Security & Admin](../bosco/security.md).

Workbench embeds an ML agent — **Bosco** — directly in the Python REPL. Chemists
and data scientists describe what they want in plain language — *"which compounds
does this model get worst?"*, *"build me a solubility model from this FeatureSet"*
— and Bosco writes the code, runs it in the live session, reads the output, and
adjusts.

That turns exploratory ML work into a conversation. And Bosco carries Workbench
conventions, so the code he produces looks like the code your team already
writes.

## One session, two drivers

The REPL and Bosco share a single session. You type Python when you want
to; you talk to Bosco when that's faster — ask him to review the code you
just wrote, show you an example, or take a task and run with it. Either way the
work lands in the same namespace, so control passes back and forth without
anything being copied or handed off.

<img src="../../images/agent_repl_chat.svg" alt="Inside the Workbench REPL, you and the ML agent both write and run code in one shared session; you direct the agent by chat — review this, show me an example, take it from here — and variables persist for both of you" style="width: 100%; min-height: 360px;">

The loop runs against the real thing: the agent reads your actual FeatureSets,
your actual model predictions, and your actual compounds, so its next
suggestion is grounded in what your data really looks like rather than a
description of it. And because it all happens in your session, it stops wherever
you want — take the dataframe it just built and keep going by hand.

## Where the work happens

Bosco runs Claude through Amazon Bedrock, so the whole loop stays inside
your own AWS account — authenticated by the Workbench IAM role you already use,
and billed on the same invoice as SageMaker. Every request is TLS-encrypted and
signed with your role's credentials, and the model itself runs on AWS
infrastructure in your region.

Most agent tooling instead reaches a model through a vendor API, which means an
account to create, a key to distribute, and a second invoice. Going through
Bedrock skips all of it:

| | Vendor-hosted model API         | Workbench on Bedrock |
|---|---------------------------------|------------------|
| Setup | New vendor account, new API key | Automatic on AWS |
| Credentials on laptops | An API key per person           | None — the AWS profile you already have |
| Cost | A separate invoice to reconcile | A line on your AWS bill |
| Adding a teammate | Issue and track another key     | They already have the role |
| Your data | Exposed to a third party        | Stays in your AWS account |

Teams that want the details — IAM roles, retention settings, PrivateLink — will
find them on the [Security & Admin](../bosco/security.md) page.

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS
and Workbench. Please contact us at
[workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or chat us
up on [Discord](https://discord.gg/WHAJuz8sw8)
