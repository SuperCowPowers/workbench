# Bosco: the Workbench ML Agent

!!! tip inline end "Background"
    Why an agent in the REPL at all: [The Workbench ML Agent](../blogs/ml_agent.md).
    Where your data goes: [AWS Bedrock Security](../aws_setup/bedrock_security.md).

Bosco is an ML engineering agent living inside the [Workbench REPL](index.md).
He writes and runs Python in **your** session — the variables he creates stay in
your namespace, and he reads your real FeatureSets, models, and predictions.

Bosco is opt-in per account. Set `ENABLE_BOSCO` in your Workbench config and
make sure Bedrock is reachable ([AWS Bedrock Setup](../aws_setup/bedrock_setup.md));
when both are true, `Bosco` appears in the REPL prompt.

## Talking to it

There is no mode to switch into. Type a question and it goes to Bosco; type
Python and it runs as Python.

```
Workbench:scp_sandbox:Bosco> what pxr models do we have?
Workbench:scp_sandbox:Bosco> models_df = models(details=True)
Workbench:scp_sandbox:Bosco> bosco compare those two on their holdout
```

The `bosco <text>` prefix forces a question when the text *is* valid Python.

**How the REPL decides:**

| You type | Where it goes |
|---|---|
| Anything that isn't valid Python | Bosco |
| Anything ending in `?` | Bosco — always |
| A word that isn't defined (`both`, `sure`, `metrics`) | Bosco, as a reply |
| Valid Python | Runs as Python |
| Magics and shell (`%time`, `!ls`) | IPython |
| `?Model` / `??Model` | IPython — docstring / source |

Object help is the **prefix** form here. A trailing `?` is never valid Python
and is nearly always a question, so `Model?` reaches Bosco and `?Model` gets
you the docstring.

## Multi-line input

- **⌥ Option+Enter** (Alt on non-Mac keyboards) or **Ctrl-J** — new line
- **Enter** — send
- **Paste** — multi-line paste lands as-is

Shift+Enter can't work out of the box: terminals send the same byte for it as
for Enter. Map it to Ctrl-J and it works.

| Terminal | Setting |
|---|---|
| iTerm2 | Settings → Keys → Key Bindings → `⇧↩` → Send Hex Code → `0x0a` |
| kitty | `map shift+enter send_text all \x0a` |
| WezTerm | `{key="Enter", mods="SHIFT", action=wezterm.action.SendString("\n")}` |
| Ghostty | `keybind = shift+enter=text:\x0a` |

## Settings

Say **"show code"** or **"hide code"** and Bosco flips it himself, or set the
attributes directly:

```python
bosco.show_code = True         # echo the code Bosco runs (default False)
bosco.effort = "medium"        # low, medium, high (default), xhigh, max
bosco.usage                    # session token counts and estimated cost_usd
```

`effort` is thinking depth per turn, not reply length — lower is faster, and the
difference only shows on questions hard enough to think about.

**Ctrl-C** interrupts at any point, mid-thought or mid-query. The conversation
stays usable, so the next question works normally.

## Sessions

Bosco can write a session report — the goal, the artifacts involved, what was
concluded, what is still open — to the Parameter Store, so anyone on the account
can pick the thread back up.

```python
recent_sessions()          # what's saved, most recent first
show_session("logd-cleanup")
show_session()             # the most recently saved, any user
```

Ask Bosco to "save a session" and it writes one. Reports are prose, not
transcripts: they name artifacts rather than restating them.

## Long-running work

Bosco is turn-based, so he cannot interrupt you when a job finishes. Batch work
is pull-based instead — `batch_jobs()` shows what is running, and anything that
completed since your last turn is handed to Bosco at the start of the next one,
so he can speak to the outcome.

## Zero data retention

Bedrock's default retention keeps inputs and outputs for AWS safety and abuse
prevention; the model provider never receives them. Setting the account
retention mode to `none` guarantees nothing is stored at all.

There is no console for this — it is an API call, and it needs admin
credentials:

```bash
AWS_PROFILE=<profile> aws bedrock put-account-data-retention --region <region> --mode none
```

Verify it took:

```bash
AWS_PROFILE=<profile> aws bedrock get-account-data-retention --region <region>
```

The setting is account-wide, not per-user. Two things to know before you flip
it: some models require per-account ZDR approval from AWS before `none` is
permitted, and any model that does not allow the mode simply becomes
unavailable to the account. The `bedrock:DataRetentionMode` condition key lets a
Service Control Policy keep anyone from loosening it afterwards.

Full context — retention modes, PrivateLink, invocation logging, and what the
model provider can and cannot see — is on the
[AWS Bedrock Security](../aws_setup/bedrock_security.md) page.

## Questions?
<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to help. Reach us at
[workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on
[Discord](https://discord.gg/WHAJuz8sw8).
