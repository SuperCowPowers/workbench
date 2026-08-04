"""Light and dark color themes for the Workbench REPL.

One theme is live at a time, chosen by the ``REPL_THEME`` config (default ``dark``)
and switchable with :func:`set_theme`. ``PALETTE``, ``colors``, and ``prompt_styles``
are **mutated in place** rather than rebound, so everything holding a reference to
them -- ``cprint``, the prompt, Bosco's markdown -- follows a switch without
re-importing.

Add a color by adding one entry to *both* palettes; the keys must match.
"""

import logging

from pygments.token import Token

log = logging.getLogger("workbench")

DEFAULT_THEME = "dark"

# Tuned for a dark terminal: light, saturated values that carry against near-black.
_DARK = {
    "lightblue": "#5f87ff",
    "lightpurple": "#af87ff",
    "lightgreen": "#87d75f",
    "darkyellow": "#ffd700",
    "orange": "#ff8700",
    "red": "#dd0000",
    "pink": "#ff87ff",
    "magenta": "#ff5fd7",
    "tan": "#d7af5f",
    "lighttan": "#d7af87",
    "yellow": "#ffff00",
    "green": "#00af00",
    "purple": "#8700af",
    "darkblue": "#5f5fff",
    "lightgrey": "#bcbcbc",
    "grey": "#808080",
    "darkgrey": "#585858",
}

# The same roles darkened for a light terminal. Not an inversion -- each value is
# picked to hold contrast against white, which the dark set's pastels do not.
_LIGHT = {
    "lightblue": "#0050c7",
    "lightpurple": "#6a3fbf",
    "lightgreen": "#3f7f1f",
    "darkyellow": "#9a7d00",
    "orange": "#c25e00",
    "red": "#c00000",
    "pink": "#b03fb0",
    "magenta": "#b0009a",
    "tan": "#8a6a1f",
    "lighttan": "#9a7550",
    "yellow": "#8a8a00",
    "green": "#007000",
    "purple": "#6a008a",
    "darkblue": "#3a3ac7",
    "lightgrey": "#6a6a6a",
    "grey": "#565656",
    "darkgrey": "#3a3a3a",
}

THEMES = {"dark": _DARK, "light": _LIGHT}

# Which palette color plays each role in the REPL prompt.
_PROMPT_ROLES = {
    Token.Workbench: "lightpurple",
    Token.Darkyellow: "darkyellow",
    Token.Lightpurple: "lightpurple",
    Token.Lightgreen: "lightgreen",
    Token.Orange: "orange",
    Token.Red: "red",
    Token.Blue: "darkblue",
    Token.Grey: "lightgrey",
}

# Live views, populated by set_theme() below. Never rebind these -- mutate in place.
PALETTE = {}
colors = {}
prompt_styles = {}

RESET = "\x1b[0m"

_current = None


def _ansi(hex_str: str) -> str:
    """Truecolor foreground escape for a hex string."""
    r, g, b = int(hex_str[1:3], 16), int(hex_str[3:5], 16), int(hex_str[5:7], 16)
    return f"\x1b[38;2;{r};{g};{b}m"


def current_theme() -> str:
    """Name of the theme in effect."""
    return _current


def set_theme(name: str) -> None:
    """Switch the REPL color theme.

    Updates the palette, the ANSI escapes, the prompt styles, and the log-level
    colors together, then restyles a running prompt if there is one.

    Args:
        name (str): "dark" or "light".

    Raises:
        ValueError: If the theme name is unknown.
    """
    global _current
    if name not in THEMES:
        raise ValueError(f"Unknown theme {name!r}; choose from {', '.join(THEMES)}")

    PALETTE.clear()
    PALETTE.update(THEMES[name])

    colors.clear()
    colors.update({key: _ansi(hex_str) for key, hex_str in PALETTE.items()})
    colors["reset"] = RESET

    prompt_styles.clear()
    prompt_styles.update({token: PALETTE[role] for token, role in _PROMPT_ROLES.items()})

    _current = name
    _set_log_theme(name)
    _restyle_prompt()


def _set_log_theme(name: str) -> None:
    """Point the log formatter at the matching theme, re-styling live handlers."""
    from workbench.utils.workbench_logging import ColoredFormatter

    ColoredFormatter.set_theme(name)
    for handler in logging.getLogger("workbench").handlers:
        if isinstance(handler.formatter, ColoredFormatter):
            handler.setFormatter(
                ColoredFormatter(
                    "%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )


def _restyle_prompt() -> None:
    """Push the new prompt styles into a running IPython shell, if there is one."""
    from IPython import get_ipython

    shell = get_ipython()
    if shell is None or not hasattr(shell, "refresh_style"):
        return
    shell.highlighting_style_overrides = dict(prompt_styles)
    shell.refresh_style()


def _configured_theme() -> str:
    """Theme named by the REPL_THEME config, falling back to the default."""
    from workbench.utils.config_manager import ConfigManager

    try:
        name = str(ConfigManager().get_config("REPL_THEME", DEFAULT_THEME)).strip().lower()
    except Exception:
        return DEFAULT_THEME
    if name not in THEMES:
        log.warning(f"Unknown REPL_THEME {name!r}; using {DEFAULT_THEME!r} ({', '.join(THEMES)})")
        return DEFAULT_THEME
    return name


set_theme(_configured_theme())
