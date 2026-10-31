# Using Bosco

> how to interact with Bosco: routing, multi-line, Shift+Enter, model and effort settings, interrupting

Read this when the user asks how to interact with you, or to change your model
or effort.

## Two ways in

```
what pxr models do we have?     # just type -- anything that isn't valid Python
bosco what models do we have    # explicit prefix, for text that IS valid Python
```

The REPL routes automatically: valid Python runs as Python, everything else
comes to you. Magics (`%time`) and shell (`!ls`) still work normally. A lone
undefined word (`both`, `yes`, `sure`, `metrics`) is treated as a reply to you,
not code.

A line ending in `?` always comes to you, so object help is the prefix form:
`?Model` for the docstring, `??Model` for the source. Say so if someone types
`Model?` and gets you instead.

## Multi-line input

- **Shift+Enter** — new line, once the terminal is set up (below)
- **Enter** — send
- **Paste** — multi-line paste lands as-is, no key needed

A terminal sends the same byte for Shift+Enter as for Enter, so nothing can
tell them apart until it's mapped to newline (hex `0x0a`). It's a one-time
setting, and it then works in every terminal program, not just here.

| Terminal | Setting |
|---|---|
| iTerm2 | Settings → Keys → Key Bindings → **+** → press `⇧↩` → Send Hex Code → `0x0a` |
| kitty | `map shift+enter send_text all \x0a` |
| Ghostty | `keybind = shift+enter=text:\x0a` |
| WezTerm | `{key="Enter", mods="SHIFT", action=wezterm.action.SendString("\n")}` |
| VS Code | `{"key": "shift+enter", "command": "workbench.action.terminal.sendSequence", "args": {"text": "\n"}, "when": "terminalFocus"}` |

## Settings

`bosco()` prints the current values; its docstring lists the options.

- **"show code"** / **"hide code"** — echo the code you run (`bosco.show_code`).
  The spoken toggle is the easier path to offer.
- **Model and effort** — the user's own settings, kept in their Workbench config.
  When they ask for a change, save it. It applies on your next round and sticks
  across sessions:

  ```python
  from workbench.utils.config_manager import ConfigManager

  ConfigManager().save_config("BOSCO_MODEL", "claude-sonnet-5")  # or claude-opus-5 (default)
  ConfigManager().save_config("BOSCO_EFFORT", "low")             # low, medium, high (default), xhigh, max
  ```

  In local mode there is no config file, so tell the user the change lasts only
  for this session.

  Effort is thinking depth per turn, not reply length. It's the main speed lever:
  lower means faster turns and fewer tool calls, and less checking against live
  data. At low effort Sonnet matches Opus's speed at about half the cost. Offer
  Sonnet at low effort when someone wants snappy turns, and suggest raising effort
  for build work or anything that should be verified against real data.

## Interrupting

**Ctrl-C** stops you at any point — mid-thought, mid-query, mid-tool. The
conversation stays usable afterwards, so the next question works normally.
