"""Multi-page PDF reports: one shared look, page helpers, and typeset equations.

A report is a different artifact from a plot. `workbench.utils.plots.hpo` and friends
return one figure answering one question on screen; this module composes a document that
gets forwarded to someone who was not in the room.

Call `theme()` once, then build one function per page, each ending in `pdf.savefig(fig)`.
Every layout helper takes the running `y` cursor and returns the new one, so a page body
reads as a chain down the page. See the `reports` guide for structure advice.

Fonts degrade gracefully: headings and body text prefer installed faces but fall back to
DejaVu Sans, and equations use matplotlib's bundled STIX, so a report renders the same on
a laptop, a Linux box, or inside a Batch container with no fonts installed.
"""

import textwrap
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager

# Softened ink rather than pure black -- three greys give hierarchy without bolding
# everything: INK for headings and numbers, BODY for prose, MUTED for labels and units.
INK = "#111827"
BODY = "#3F4651"
MUTED = "#9AA1AC"
RULE = "#E4E7EB"
HAIRLINE = "#F0F2F4"

# Accents for groups, cohorts, or model families. Used in order; a group keeps its colour
# on every page so a reader can track it without reading the label.
ACCENTS = ["#0F766E", "#1D4ED8", "#B45309", "#BE123C", "#6D28D9", "#047857"]

# Page margins in figure coordinates. Every helper honours these, so text columns and
# rules line up across pages without per-page arithmetic.
L, R = 0.068, 0.932

# Preferred faces, best first. DejaVu Sans ships with matplotlib, so the last entry is
# always available -- these lists never need a guard.
HEADING_PREFS = ["Inter SemiBold", "Avenir Next", "Helvetica Neue", "Roboto", "Lato", "DejaVu Sans"]
BODY_PREFS = ["Inter", "Helvetica Neue", "Helvetica", "Roboto", "Open Sans", "Arial", "DejaVu Sans"]

# Nice text faces carry small character sets and matplotlib substitutes a missing glyph
# into a visible box, silently. These are the ones that actually bite; text helpers
# substitute them automatically.
ASCII_FALLBACKS = {"\u2192": "->", "\u2190": "<-", "\u21d2": "=>", "\u2212": "-"}

# Equations worth having typeset rather than faked with carets and slashes. mathtext
# parses all of these; add your own as raw strings wrapped in $...$.
EQ = {
    "bateman": r"$C(t)\;=\;\frac{F\cdot D\cdot k_a\,k_e}{CL\,(k_a-k_e)}\;" r"\left(e^{-k_e t}-e^{-k_a t}\right)$",
    "tmax": r"$t_{max}\;=\;\dfrac{\ln(k_a/k_e)}{k_a-k_e}$",
    "auc": r"$AUC\;=\;\dfrac{F\cdot D}{CL}$",
    "gmfe": r"$GMFE\;=\;10^{\,\mathrm{median}\,\left|\log_{10}\left(C_{pred}/C_{obs}\right)\right|}$",
    "ceiling": r"$F_{max}\;=\;100\left(1-\dfrac{CL}{Q_h}\right)$",
    "ke_vd": r"$k_e\;=\;\dfrac{CL\cdot 0.06}{V_d}$",
    "ke_lz": r"$k_e\;=\;\lambda_z\quad\mathrm{(terminal\ slope)}$",
    "r2": r"$R^2\;=\;1-\dfrac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$",
    "rmse": r"$RMSE\;=\;\sqrt{\dfrac{1}{n}\sum(y-\hat{y})^2}$",
}


def pick_font(prefs: Sequence[str], fallback: str = "DejaVu Sans") -> str:
    """First font in `prefs` that is actually installed, else `fallback`.

    Never assume a face exists. Avenir Next and Helvetica Neue are macOS-only, so a
    report that hardcodes them looks right on one laptop and quietly degrades everywhere
    else -- better to choose explicitly and know what you got.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((p for p in prefs if p in available), fallback)


HEADING_FONT = pick_font(HEADING_PREFS)
BODY_FONT = pick_font(BODY_PREFS)


def theme(heading: Optional[str] = None, body: Optional[str] = None) -> dict:
    """Apply the report look to rcParams. Call once, before building pages.

    Equations are set in **STIX** (`mathtext.fontset="stix"`), which matplotlib bundles --
    so maths renders identically on every machine with no font installs, and a serif
    equation against a sans body is the normal journal pairing rather than a mismatch.

    `pdf.fonttype=42` embeds TrueType, which is what makes the text in the finished PDF
    selectable and searchable; matplotlib's Type 3 default is neither.

    Returns the chosen face names so a caller can report or override them.
    """
    global HEADING_FONT, BODY_FONT
    HEADING_FONT = heading or pick_font(HEADING_PREFS)
    BODY_FONT = body or pick_font(BODY_PREFS)
    plt.rcParams.update(
        {
            "font.family": BODY_FONT,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": BODY,
            "axes.edgecolor": "#C9CED6",
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelsize": 9.5,
            "figure.facecolor": "white",
        }
    )
    return {"heading": HEADING_FONT, "body": BODY_FONT, "math": "stix"}


def _clean(text: str) -> str:
    """Swap glyphs the nice faces lack for ASCII. Math (`$...$`) is left alone."""
    if not isinstance(text, str) or "$" in text:
        return text
    for bad, good in ASCII_FALLBACKS.items():
        text = text.replace(bad, good)
    return text


def new_page(size: Tuple[float, float] = (8.5, 11)) -> plt.Figure:
    """Blank portrait page. Compose in figure coordinates, not subplots."""
    return plt.figure(figsize=size, facecolor="white")


def rule(fig: plt.Figure, y: float, x0: float = L, x1: float = R, c: str = RULE, lw: float = 0.9) -> None:
    """Hairline horizontal rule. A visible table border is a 1990s table border."""
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=c, lw=lw, transform=fig.transFigure))


def head(fig: plt.Figure, kicker: str, title: str, sub: Optional[str] = None) -> float:
    """Page header: letterspaced kicker, large title, optional standfirst, rule.

    Returns the `y` to start the body at.
    """
    fig.text(L, 0.962, "  ".join(_clean(kicker).upper()), fontname=HEADING_FONT, fontsize=7.6, color=MUTED)
    fig.text(L, 0.936, _clean(title), fontname=HEADING_FONT, fontsize=25, color=INK, va="top")
    y = 0.906
    if sub:
        for line in textwrap.wrap(_clean(sub), 104):
            fig.text(L, y, line, fontsize=10.2, color=BODY, va="top")
            y -= 0.0165
    rule(fig, y - 0.008)
    return y - 0.030


def sec(fig: plt.Figure, y: float, label: str, color: str = INK) -> float:
    """Small uppercase section label. Give every text block one of these."""
    fig.text(L, y, _clean(label).upper(), fontname=HEADING_FONT, fontsize=9.2, color=color, va="top")
    return y - 0.021


def para(
    fig: plt.Figure,
    y: float,
    text: str,
    size: float = 9.5,
    width: int = 112,
    color: str = BODY,
    x: float = L,
    lead: float = 0.0155,
    gap: float = 0.010,
) -> float:
    """Wrapped prose. Blank lines in `text` become paragraph breaks.

    Keep any one block to about four lines -- a full page of paragraphs is skipped by
    every reader who matters.
    """
    for block in _clean(text).split("\n"):
        if not block.strip():
            y -= gap
            continue
        for line in textwrap.wrap(block, width) or [""]:
            fig.text(x, y, line, fontsize=size, color=color, va="top")
            y -= lead
        y -= gap * 0.4
    return y


def equation(
    fig: plt.Figure,
    y: float,
    tex: str,
    size: float = 15,
    color: str = INK,
    x: float = 0.5,
    ha: str = "center",
    lead: float = 2.8,
) -> float:
    """Display equation, centred, set in STIX.

    `tex` is mathtext -- a raw string wrapped in `$...$`, e.g. `EQ["bateman"]`. Use
    `\\dfrac` for a display-size fraction bar and `\\frac` for an inline one.

    Height is estimated from the point size (`lead` multiples of a line), since a
    stacked fraction is taller than its font size; nudge `lead` if a page gets tight.
    """
    fig.text(x, y, tex, fontsize=size, color=color, ha=ha, va="top")
    return y - lead * size / (72 * fig.get_figheight())


def table(
    fig: plt.Figure,
    y: float,
    headers: Sequence[str],
    rows: Sequence[Sequence],
    xs: Sequence[float],
    aligns: Optional[Sequence[str]] = None,
    size: float = 8.8,
    hl_row: Optional[int] = None,
    lead: float = 0.0168,
) -> float:
    """Positioned-text table with hairline rules. `xs` are column x positions.

    Drawn as text rather than as a monospace block on purpose: ASCII tables are the main
    thing that makes a PDF look like a terminal dump. `hl_row` sets one row in the
    heading face, for the variant you actually adopted.
    """
    aligns = list(aligns or ["left"] * len(headers))
    for h, x, a in zip(headers, xs, aligns):
        fig.text(x, y, _clean(h).upper(), fontsize=6.9, color=MUTED, va="top", fontname=HEADING_FONT, ha=a)
    y -= 0.014
    rule(fig, y + 0.004, c="#D6DADF")
    y -= 0.006
    for i, row in enumerate(rows):
        hot = hl_row is not None and i == hl_row
        for v, x, a in zip(row, xs, aligns):
            fig.text(
                x,
                y,
                _clean(str(v)),
                fontsize=size,
                color=INK if hot or hl_row is None else BODY,
                va="top",
                ha=a,
                fontname=HEADING_FONT if hot else BODY_FONT,
            )
        y -= lead
        if i < len(rows) - 1:
            rule(fig, y + 0.0045, c=HAIRLINE)
    return y - 0.006


def stat_row(
    fig: plt.Figure, x: float, y: float, pairs: Sequence[Tuple[str, str]], size: float = 8.6, dx: float = 0.082
) -> float:
    """Row of label-over-value stats. The cheapest way to put numbers near prose."""
    for i, (k, v) in enumerate(pairs):
        fig.text(x + i * dx, y, _clean(k).upper(), fontsize=6.6, color=MUTED, va="top", fontname=HEADING_FONT)
        fig.text(x + i * dx, y - 0.0125, _clean(str(v)), fontsize=size, color=INK, va="top", fontname=HEADING_FONT)
    return y - 0.030


def footer(fig: plt.Figure, left: str, page: Optional[object] = None) -> None:
    """Quiet page furniture. Cheap, and it makes a loose printed page identifiable."""
    fig.text(L, 0.028, _clean(left), fontsize=7.6, color=MUTED)
    if page is not None:
        fig.text(R, 0.028, str(page), fontsize=7.6, color=MUTED, ha="right")


def audit_glyphs(build: Callable, *args, **kwargs) -> List[str]:
    """Run a report build and return the glyphs no font could supply.

    matplotlib substitutes a missing character into a visible box without raising, so a
    document has to be audited rather than eyeballed. An empty list means clean; anything
    else gets an entry in `ASCII_FALLBACKS` or a different character.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build(*args, **kwargs)
        return sorted({str(w.message) for w in caught if "Glyph" in str(w.message)})
