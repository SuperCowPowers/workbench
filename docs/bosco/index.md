# Bosco: the Workbench ML Agent

!!! tip inline end "Background"
    Why an agent in the REPL at all: [The Workbench ML Agent](../blogs/ml_agent.md).
    Where your data goes: [Security & Admin](security.md).

Bosco is an ML engineering agent living inside the
[Workbench REPL](../repl/index.md). He writes and runs Python in **your**
session — the variables he creates stay in your namespace, and he reads your
real FeatureSets, models, and predictions.

How Bosco reaches a model depends on whether you have an AWS account.

**With one**, Bosco is opt-in per account: set `ENABLE_BOSCO` in your Workbench
config and make sure Bedrock is reachable ([Security & Admin](security.md)).
Prompts stay inside your account.

**Without one**, the REPL comes up in [local mode](../local/index.md) and there is
no Bedrock to reach, so Bosco uses an Anthropic API key instead:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
workbench
```

Either way `Bosco` appears in the REPL prompt once a model is reachable, and
`status` names the path it took.

## Talking to it

There is no mode to switch into. Type a question and it goes to Bosco; type
Python and it runs as Python.

```
Workbench:scp_sandbox:Bosco> what pxr models do we have?
Workbench:scp_sandbox:Bosco> models_df = models(details=True)
Workbench:scp_sandbox:Bosco> compare those two on their holdout
```

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
can pick the thread back up. In local mode they are files under
`WORKBENCH_LOCAL_PATH` instead, private to that machine.

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

## Where your data goes

Bosco's prompts carry real data by design, and they travel over Bedrock inside
your own AWS account — same IAM, same CloudTrail, same bill. Our standard Bedrock
setup turns on ZDR (Zero Data Retention), so no data is stored.

The full picture — enabling Bosco, retention modes, zero data retention,
invocation logging, auditing, and PrivateLink — is on
[Security & Admin](security.md).

## Multi-line input

- **Shift+Enter** — new line (one-time terminal setup, below)
- **Enter** — send
- **Paste** — multi-line paste lands as-is

### Terminal setup for Shift+Enter

A terminal sends the same byte for Shift+Enter as it does for Enter, so no
program can tell them apart. Map Shift+Enter to newline (hex `0x0a`) once in
your terminal and it works everywhere, including here.

**iTerm2**

1. Settings → Keys → Key Bindings → **+**
2. Click the shortcut field and press **⇧↩**
3. Action: **Send Hex Code**, value `0x0a`

**Other terminals** — add to the config file:

| Terminal | Setting |
|---|---|
| kitty | `map shift+enter send_text all \x0a` |
| Ghostty | `keybind = shift+enter=text:\x0a` |
| WezTerm | `{key="Enter", mods="SHIFT", action=wezterm.action.SendString("\n")}` |
| VS Code | `{"key": "shift+enter", "command": "workbench.action.terminal.sendSequence", "args": {"text": "\n"}, "when": "terminalFocus"}` |

## Questions?
<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to help. Reach us at
[workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on
[Discord](https://discord.gg/WHAJuz8sw8).
