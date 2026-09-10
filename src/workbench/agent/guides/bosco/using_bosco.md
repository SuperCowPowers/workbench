# Using Bosco

> how to interact with Bosco: routing, multi-line, Shift+Enter, settings, interrupting

Read this when the user asks how to interact with you.

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

Attributes on `bosco` — read its docstring for the current values and levels.

- **"show code"** / **"hide code"** — echo the code you run (`bosco.show_code`).
  The spoken toggle is the easier path to offer.
- **`bosco.effort`** — thinking depth per turn, not reply length. No spoken
  toggle; lower is faster, and it only shows on questions hard enough to think
  about.

## Interrupting

**Ctrl-C** stops you at any point — mid-thought, mid-query, mid-tool. The
conversation stays usable afterwards, so the next question works normally.
