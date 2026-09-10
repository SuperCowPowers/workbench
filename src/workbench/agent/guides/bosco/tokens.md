# How Bosco Manages Tokens

> how Bosco manages token usage, context, and cost

Read this when the user asks about token usage, cost, or context.

Every LLM call is billed on input (everything sent) plus output (what's
generated). In an agentic loop the input is resent on *every* round, so the same
context gets paid for repeatedly. Bosco's design is built around that.

## Lazy guides

Only `general.md` is always loaded; the rest are read on demand via `read_guide`.
So the fixed per-call overhead is the frame plus one short guide, not the whole
library — which is well over an order of magnitude larger. (For the current
numbers, measure: `guides/` on disk, ~4 chars per token.)

This is the main lever, and it shapes where a rule belongs:

- Applies **every turn** regardless of task → `general.md`, billed on every call
- Only matters **while doing a specific thing** → a guide, nearly free until read

Adding a guide costs almost nothing. Adding to `general.md` costs forever.

## Prompt caching

Each round resends tools + system + the whole conversation, and that prefix is
identical every time. Bosco sets a rolling cache breakpoint on the newest
message, so the prefix returns as a cache read (~10% of normal cost) rather than
being re-billed.

It holds *across* turns, not just within one, so the savings compound over a
session.

## Bounded growth

- **History** is capped (~50k tokens), dropping oldest exchanges. It only ever
  cuts at a real user prompt so a `tool_use` block is never split from its
  `tool_result`. The cap is a soft target — it will exceed it rather than
  corrupt the conversation.
- **Tool output** is truncated at 4000 chars. Results live in history and are
  resent every later round, so a large dump is paid many times over.
- **Tool rounds** stop at 25 per turn, as a runaway guard.

## What the session actually cost

```python
bosco.usage    # input/output/cache counts, calls, and estimated cost_usd
```

Asking questions, reading guides, and long conversations are cheap — caching
absorbs them. If the user wants to reduce usage further, the honest answer is
that the architecture already handles the big levers; the rest is Bosco's own
discipline (`general` → Working style).
