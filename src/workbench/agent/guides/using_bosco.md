# Using Bosco

> how to interact with Bosco: routing, multi-line, Shift+Enter, interrupting

Read this when the user asks how to interact with you.

## Two ways in

```
what pxr models do we have?     # just type -- anything that isn't valid Python
bosco what models do we have    # explicit prefix, for text that IS valid Python
```

The REPL routes automatically: valid Python runs as Python, everything else
comes to you. Magics (`%time`), shell (`!ls`), and object help (`Model?`) still
work normally. A lone undefined word (`both`, `yes`, `metrics`) is treated as a
reply to you, not code.

## Multi-line input

- **⌥ Option+Enter** (labeled Alt on non-Mac keyboards) or **Ctrl-J** — new line
- **Enter** — send
- **Paste** — multi-line paste lands as-is, no key needed

### Shift+Enter

It doesn't work out of the box, and that's a terminal limitation rather than a
Workbench one: in the legacy terminal protocol Shift+Enter sends the same byte
as Enter, so nothing can tell them apart. (The CSI-u protocol that *can* encode
it has to be negotiated by the application, and the REPL doesn't do that.)

Map it to Ctrl-J in the terminal and it works:

| Terminal | Setting |
|---|---|
| iTerm2 | Settings → Keys → Key Bindings → `⇧↩` → Send Hex Code → `0x0a` |
| kitty | `map shift+enter send_text all \x0a` |
| WezTerm | `{key="Enter", mods="SHIFT", action=wezterm.action.SendString("\n")}` |
| Ghostty | `keybind = shift+enter=text:\x0a` |

## Showing the code

Code echo is off by default:

```python
bosco.show_code = True    # echo the code Bosco runs
bosco.show_code = False   # answers only
```

## Interrupting

**Ctrl-C** stops you at any point — mid-thought, mid-query, mid-tool. The
conversation stays usable afterwards, so the next question works normally.

## What persists

Variables you create stay in the user's session after the turn ends, and the
conversation carries across turns — so follow-up questions have context.
