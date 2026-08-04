"""Terminal color output for the Workbench REPL.

Palette values and theme switching live in `repl_themes`; this is the rendering
on top of them -- colored text and markdown.
"""

from workbench.utils import repl_themes

# ---------------------------------------------------------------------------
# REPL terminal palette
#
# The values live in `repl_themes`, which owns the light and dark sets and swaps
# between them. These are the live dicts, mutated in place on a theme switch, so
# everything downstream follows without re-importing:
#   - cprint() / colors  -> truecolor ANSI escapes for terminal text
#   - render_markdown()  -> the rich theme for Bosco's replies
#
# Truecolor (24-bit) is emitted directly, not a 256-palette index, so a color
# renders as its exact RGB regardless of how a terminal themes its 256 slots.
# ---------------------------------------------------------------------------

# Pygments style for fenced code blocks. Any pygments theme name works.
CODE_THEME = "monokai"

PALETTE = repl_themes.PALETTE
colors = repl_themes.colors


def cprint(*args):
    """Print text in color. Either a single color and text, or a list of
    color-text pairs.

    Example: cprint('red', 'Hello') or cprint(['red', 'Hello', 'green', 'World'])
    """
    if isinstance(args[0], list):
        args = args[0]
    for i in range(0, len(args), 2):
        print(f"{colors[args[i]]}{args[i + 1]}{colors['reset']}", end=" ")
    print()


def render_markdown(text: str) -> None:
    """Render markdown (tables, bold, headers, lists) in the terminal.

    Used for the Bosco agent's replies and session reports: prose in Bosco's blue,
    code and bold in green with no background box, headings in Workbench's purple.
    Falls back to plain colored text if rich is unavailable.

    Follows the active theme, since the palette is read per call.
    """
    try:
        from rich.console import Console
        from rich.markdown import Markdown, CodeBlock, Heading
        from rich.syntax import Syntax
        from rich.theme import Theme
    except ImportError:
        cprint("lightblue", text)
        return

    class _NoBoxCodeBlock(CodeBlock):
        """Fenced code without rich's full-width background box."""

        def __rich_console__(self, console, options):
            yield Syntax(
                str(self.text).rstrip(),
                self.lexer_name,
                theme=self.theme,
                word_wrap=True,
                padding=0,
                background_color="default",
            )

    class _LeftHeading(Heading):
        """Headings all flush left; rich centers h1 by default."""

        LEVEL_ALIGN = {}

    theme = Theme(
        {
            "markdown.text": PALETTE["lightblue"],  # prose in Bosco's blue
            "markdown.paragraph": PALETTE["lightblue"],
            "markdown.code": f"bold {PALETTE['lightgreen']}",  # inline code: green, no bg box
            "markdown.strong": f"bold {PALETTE['lightgreen']}",  # bold: green, no bg box
            # Headings step down in weight, all in Workbench's purple
            "markdown.h1": f"bold underline {PALETTE['lightpurple']}",
            "markdown.h2": f"bold {PALETTE['lightpurple']}",
            "markdown.h3": PALETTE["lightpurple"],
            "markdown.h4": f"italic {PALETTE['lightpurple']}",
            # Everything else rich would leave on raw 16-color ANSI
            "markdown.link": f"underline {PALETTE['lightblue']}",
            "markdown.link_url": f"underline {PALETTE['darkblue']}",
            "markdown.item.bullet": f"bold {PALETTE['lightpurple']}",
            "markdown.item.number": PALETTE["darkblue"],
            "markdown.block_quote": PALETTE["tan"],
            "markdown.table.border": PALETTE["darkblue"],
            "markdown.table.header": f"bold {PALETTE['lightpurple']}",
        }
    )
    # Fenced blocks render through a Syntax object, so they take a pygments theme
    # rather than the style map above.
    markdown = Markdown(text, code_theme=CODE_THEME)
    markdown.elements["fence"] = _NoBoxCodeBlock
    markdown.elements["code_block"] = _NoBoxCodeBlock
    markdown.elements["heading_open"] = _LeftHeading
    Console(theme=theme).print(markdown)
