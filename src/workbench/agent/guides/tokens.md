# How Bosco Manages Tokens

> how Bosco manages token usage, context, and cost

Read this when the user asks about token usage, cost, or context.

Every LLM call is billed on input (everything sent) plus output (what's
generated). In an agentic loop the input is resent on *every* round, so the same
context gets paid for repeatedly. Bosco's design is built around that.

## Fixed cost per call

| | tokens |
|---|---|
| System prompt (frame + `general.md`) | ~700 |
| Tool schemas | ~197 |
| **All guides, if they were always loaded** | **~10,900 (avoided)** |

## Lazy guides

The guides total ~10.9k tokens across 15 files, but only `general.md` is
always loaded. The rest are read on demand via `read_guide`, so a typical call
carries ~900 tokens of fixed overhead instead of ~11k.

This is the main lever, and it shapes where a rule belongs:

- Applies **every turn** regardless of task → `general.md`, billed on every call
- Only matters **while doing a specific thing** → a guide, nearly free until read

Adding a guide costs almost nothing. Adding to `general.md` costs forever.

## Prompt caching

Each round resends tools + system + the whole conversation, and that prefix is
identical every time. Bosco sets a rolling cache breakpoint on the newest
message, so the prefix returns as a cache read (~10% of normal cost) rather than
being re-billed.

Measured: uncached input drops to ~2 tokens per call, with ~87% cost reduction
by the third call — and it holds *across* turns, not just within one, so savings
compound over a session.

## Bounded growth

- **History** is capped (~50k tokens), dropping oldest exchanges. It only ever
  cuts at a real user prompt so a `tool_use` block is never split from its
  `tool_result`. The cap is a soft target — it will exceed it rather than
  corrupt the conversation.
- **Tool output** is truncated at 4000 chars. Results live in history and are
  resent every later round, so a large dump is paid many times over. Hitting the
  cap usually means the filtering belonged in the query.
- **Tool rounds** stop at 25 per turn, as a runaway guard.

## What this means in practice

Cheap: asking questions, reading guides, long conversations (caching absorbs
them).

Expensive: dumping large DataFrames into tool output instead of filtering in
`query()`, or re-running the same expensive call rather than reusing a variable
already in the session.

If the user wants to reduce usage further, the honest answer is that the
architecture already handles the big levers — the remaining discipline is on
Bosco's side: filter in the query, don't print whole DataFrames, and reuse
handles that are already in the namespace.
