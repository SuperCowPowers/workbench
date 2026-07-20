# The Workbench ML Agent
!!! tip inline end "Already Using Workbench?"
    The ML agent ships with the [Workbench REPL](../repl/index.md). Model access
    is a one-time setup — see [AWS Bedrock Setup](../aws_setup/bedrock_setup.md)
    and [AWS Bedrock Security](../aws_setup/bedrock_security.md).

Workbench embeds an ML agent directly in the Python REPL. Chemists and data
scientists describe what they want in plain language — *"which compounds does
this model get worst?"*, *"build me a solubility model from this FeatureSet"* —
and the agent writes the code, runs it in the live session, reads the output,
and adjusts. Every variable it creates stays in the namespace, so you can pick
up where it left off and keep working by hand.

That turns exploratory ML work into a conversation. No context switching to a
browser, no copying dataframes into a chat window, no translating a question
into API calls before you can ask it. The agent also carries Workbench
conventions, so the code it produces looks like the code your team already
writes.

## One session, two drivers

The REPL and the agent share a single session. You type Python when you want
to; you talk to the agent when that's faster — ask it to review the code you
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

The agent runs Claude through Amazon Bedrock, so the whole loop stays inside
your own AWS account — authenticated by the Workbench IAM role you already use,
and billed on the same invoice as SageMaker. Every request is TLS-encrypted and
signed with your role's credentials, and the model itself runs on AWS
infrastructure in your region.

Most agent tooling instead reaches a model through a vendor API, which means an
account to create, a key to distribute, and a second invoice. Going through
Bedrock skips all of it:

| | Vendor-hosted model API | Workbench on Bedrock |
|---|---|---|
| Setup | New vendor account, new API key | Enable a model in the console |
| Credentials on laptops | An API key per person | None — the AWS profile you already have |
| Cost | A separate invoice to reconcile | A line on your AWS bill |
| Adding a teammate | Issue and track another key | They already have the role |
| Your data | Leaves for a third party | Stays in your AWS account |

Teams that want the details — IAM roles, retention settings, PrivateLink — will
find them on the [AWS Bedrock Security](../aws_setup/bedrock_security.md) page.

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS
and Workbench. Please contact us at
[workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or chat us
up on [Discord](https://discord.gg/WHAJuz8sw8)
