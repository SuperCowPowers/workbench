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

import logging
import textwrap
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages

log = logging.getLogger("workbench")

# Softened ink rather than pure black -- three greys give hierarchy without bolding
# everything: INK for headings and numbers, BODY for prose, MUTED for labels and units.
INK = "#111827"
BODY = "#3F4651"
MUTED = "#9AA1AC"
RULE = "#E4E7EB"
HAIRLINE = "#F0F2F4"

# The fill of a panel: a card, or any block set off from the page
PANEL = "#F6F7F9"

# Accents for groups, cohorts, or model families. Used in order; a group keeps its colour
# on every page so a reader can track it without reading the label.
ACCENTS = ["#0F766E", "#1D4ED8", "#B45309", "#BE123C", "#6D28D9", "#047857"]

# A table too wide for the page is set smaller, to no less than MIN_TABLE_SIZE points, and
# then has its longest cells trimmed, to no fewer than MIN_CELL_CHARS characters
MIN_TABLE_SIZE, MIN_CELL_CHARS, TRIM_STEPS = 6.8, 8, 2000

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
    "gmfe": r"$GMFE\;=\;10^{\,\mathrm{mean}\,\left|\log_{10}\left(pred/obs\right)\right|}$",
    "afe": r"$AFE\;=\;10^{\,\mathrm{mean}\,\log_{10}\left(pred/obs\right)}$",
    "ceiling": r"$F_{max}\;=\;100\left(1-\dfrac{CL}{Q_h}\right)$",
    "ke_vss": r"$k_e\;=\;\dfrac{CL\cdot 0.06}{V_{ss}}$",
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

# Headings are set bold. A face that ships several weights resolves to its regular one unless told.
HEADING_WEIGHT = "bold"


def _weight(fontname: str) -> str:
    """The weight text in a face is drawn at: the heading face is bold, any other regular."""
    return HEADING_WEIGHT if fontname == HEADING_FONT else "normal"


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


@lru_cache(maxsize=4096)
def _measured(text: str, size: float, fontname: str, figw: float) -> float:
    """Rendered width of a string in figure-fraction units. Cached -- it is a real render."""
    fig = plt.figure(figsize=(figw, 11))
    canvas = FigureCanvasAgg(fig)
    t = fig.text(0, 0, text, fontsize=size, fontname=fontname, fontweight=_weight(fontname))
    width = t.get_window_extent(renderer=canvas.get_renderer()).width / (fig.dpi * figw)
    plt.close(fig)
    return width


def text_width(text: str, size: float = 9.5, font: Optional[str] = None, figw: float = 8.5) -> float:
    """How wide a string will actually be, in figure-fraction units.

    Column positions should be computed from this rather than guessed. Eyeballing is how
    a label like "MEDIAN GMFE" ends up 48.6pt wide in a 36.7pt column -- it looks fine
    with short labels and collides the moment the data changes.
    """
    return _measured(text, size, font or BODY_FONT, figw)


def new_page(size: Tuple[float, float] = (8.5, 11)) -> plt.Figure:
    """Blank portrait page. Compose in figure coordinates, not subplots."""
    return plt.figure(figsize=size, facecolor="white")


def rule(fig: plt.Figure, y: float, x0: float = L, x1: float = R, c: str = RULE, lw: float = 0.9) -> None:
    """Hairline horizontal rule. A visible table border is a 1990s table border."""
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=c, lw=lw, transform=fig.transFigure))


def panel(fig: plt.Figure, x0: float, y0: float, x1: float, y1: float, fill: str = PANEL, edge: str = RULE) -> None:
    """Rounded panel behind a block of content, in figure coordinates.

    Args:
        fig (plt.Figure): The page.
        x0 (float): Left edge.
        y0 (float): Bottom edge.
        x1 (float): Right edge.
        y1 (float): Top edge.
        fill (str): Fill colour.
        edge (str): Outline colour.
    """
    width, height = fig.get_size_inches()
    box = FancyBboxPatch(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.012",
        mutation_aspect=width / height,
        fc=fill,
        ec=edge,
        lw=0.8,
        transform=fig.transFigure,
        zorder=-1,
    )
    fig.add_artist(box)


def badge(fig: plt.Figure, x: float, y: float, label: str, color: str) -> float:
    """A filled pill with a word in it, for a status or a severity. Returns its right edge.

    Args:
        fig (plt.Figure): The page.
        x (float): Left edge.
        y (float): Top edge.
        label (str): The word, set in capitals.
        color (str): Fill colour; the word is white.
    """
    label, size, height = _clean(label).upper(), 6.4, 0.0152
    width = text_width(label, size, HEADING_FONT) + 0.016
    page_width, page_height = fig.get_size_inches()
    pill = FancyBboxPatch(
        (x, y - height),
        width,
        height,
        boxstyle="round,pad=0,rounding_size=0.006",
        mutation_aspect=page_width / page_height,
        fc=color,
        ec="none",
        transform=fig.transFigure,
    )
    fig.add_artist(pill)
    middle = y - height / 2 - 0.0004
    fig.text(x + width / 2, middle, label, fontsize=size, color="white", ha="center", va="center").set(
        fontname=HEADING_FONT, fontweight=HEADING_WEIGHT
    )
    return x + width


def navigate(
    path: Union[str, Path],
    links: Sequence[Tuple[int, Tuple[float, float, float, float], int]] = (),
    bookmarks: Sequence[Tuple[str, int]] = (),
    title: Optional[str] = None,
) -> None:
    """Give a written PDF its navigation: regions that jump to a page, bookmarks, and a title.

    Args:
        path (Union[str, Path]): The PDF, rewritten in place.
        links (Sequence[Tuple[int, Tuple[float, float, float, float], int]]): For each link, the
            page it sits on, its (x0, y0, x1, y1) box in figure coordinates, and the page it opens.
        bookmarks (Sequence[Tuple[str, int]]): The reader's outline: each entry's text and the
            page it opens. A PDF with bookmarks opens with the outline showing.
        title (Optional[str]): The document title a reader shows in its window and search.

    Pages count from 0.
    """
    from pypdf import PdfReader, PdfWriter
    from pypdf.annotations import Link

    writer = PdfWriter(clone_from=PdfReader(path))
    for page, (x0, y0, x1, y1), target in links:
        width, height = float(writer.pages[page].mediabox.width), float(writer.pages[page].mediabox.height)
        rect = (x0 * width, y0 * height, x1 * width, y1 * height)
        writer.add_annotation(page, Link(rect=rect, border=[0, 0, 0], target_page_index=target))
    for text, page in bookmarks:
        writer.add_outline_item(text, page)
    if bookmarks:
        writer.page_mode = "/UseOutlines"
    if title:
        writer.add_metadata({"/Title": title})
    writer.write(path)


@lru_cache(maxsize=2048)
def _measured_height(text: str, size: float, fontname: str, figh: float) -> float:
    """Rendered height of a string in figure-fraction units, descenders included."""
    fig = plt.figure(figsize=(8.5, figh))
    canvas = FigureCanvasAgg(fig)
    t = fig.text(0, 0.5, text, fontsize=size, fontname=fontname, fontweight=_weight(fontname))
    height = t.get_window_extent(renderer=canvas.get_renderer()).height / (fig.dpi * figh)
    plt.close(fig)
    return height


@lru_cache(maxsize=512)
def _baseline_drop(size: float, fontname: str, figh: float) -> float:
    """How far below its top a top-aligned line's baseline sits, in figure-fraction units.

    A line's box runs from its tallest ascender to its deepest descender, so the baseline is
    the box's foot raised by the face's descent.
    """
    fig = plt.figure(figsize=(8.5, figh))
    renderer = FigureCanvasAgg(fig).get_renderer()
    t = fig.text(0, 0.5, "lp", fontsize=size, fontname=fontname, fontweight=_weight(fontname), va="top")
    descent = renderer.get_text_width_height_descent("lp", t.get_fontproperties(), False)[2]
    foot = (t.get_window_extent(renderer=renderer).y0 + descent) / (fig.dpi * figh)
    plt.close(fig)
    return 0.5 - foot


def text_height(text: str, size: float = 9.5, font: Optional[str] = None, figh: float = 11.0) -> float:
    """How tall a string will actually be, in figure-fraction units.

    Stack things with this rather than a constant: a title with descenders is taller than one
    without, so a fixed gap under it collides for some titles and gaps for others.
    """
    return _measured_height(text, size, font or BODY_FONT, figh)


def head(fig: plt.Figure, kicker: str, title: str, sub: Optional[str] = None, tag: Optional[str] = None) -> float:
    """Page header: letterspaced kicker, large title, optional standfirst, rule.

    The standfirst sits under the title's *measured* bottom, so a title with descenders does
    not crowd it and one without does not leave a hole. `tag` is a word set small, grey and in
    capitals after the title on its baseline, e.g. "Continued". Returns the `y` to start the
    body at.
    """
    title, size = _clean(title), 25.0
    fig.text(
        L,
        0.962,
        "  ".join(_clean(kicker).upper()),
        fontname=HEADING_FONT,
        fontweight=HEADING_WEIGHT,
        fontsize=7.6,
        color=MUTED,
    )
    fig.text(L, 0.936, title, fontname=HEADING_FONT, fontweight=HEADING_WEIGHT, fontsize=size, color=INK, va="top")
    if tag:
        baseline = 0.936 - _baseline_drop(size, HEADING_FONT, 11.0)
        after = L + text_width(title, size, HEADING_FONT) + 0.014
        fig.text(after, baseline, _clean(tag).upper(), fontsize=7.6, color=MUTED, va="baseline").set(
            fontname=HEADING_FONT, fontweight=HEADING_WEIGHT
        )
    y = 0.936 - _measured_height(title, size, HEADING_FONT, 11.0) - 0.007
    if sub:
        for line in textwrap.wrap(_clean(sub), 104):
            fig.text(L, y, line, fontsize=10.2, color=BODY, va="top")
            y -= 0.0165
    rule(fig, y - 0.008)
    return y - 0.030


def sec(fig: plt.Figure, y: float, label: str, color: str = INK) -> float:
    """Small uppercase section label. Give every text block one of these."""
    fig.text(
        L,
        y,
        _clean(label).upper(),
        fontname=HEADING_FONT,
        fontweight=HEADING_WEIGHT,
        fontsize=9.2,
        color=color,
        va="top",
    )
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


def _is_num(v) -> bool:
    """Does this cell read as a number? Decides alignment without being told."""
    s = str(v).strip().replace(",", "").rstrip("%").lstrip("<>~=")
    if s in ("", "--", "n/a", "N/A"):
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def table(
    fig: plt.Figure,
    y: float,
    headers: Sequence[str],
    rows: Sequence[Sequence],
    aligns: Optional[Sequence[str]] = None,
    size: float = 8.8,
    hl_row: Optional[int] = None,
    lead: float = 0.0172,
    x0: float = L,
    x1: float = R,
    gutter: float = 0.018,
    row_rules: bool = True,
    header_color: str = ACCENTS[1],
) -> float:
    """Measured table, ranged across the text column. Returns the `y` to continue at.

    Column positions are **derived, never passed**: every header and cell is measured, so a
    column is as wide as its widest content and the slack spreads evenly across the gaps --
    the label column sits at `x0`, the last column lands flush at `x1`. A hand-placed column
    is the most reliable way to bury a number under its neighbour, so there is no way to
    hand-place one here.

    A table wider than the column is made to fit rather than run off the page: the body is
    set smaller, down to a floor, and then the longest text of the widest column is trimmed
    with an ellipsis until it fits. It warns only if that is still not enough.

    Alignment is inferred -- numeric columns range right, text ranges left -- and `aligns`
    overrides per column when inference is wrong (an id that looks like a float, say).
    `hl_row` sets one row in the heading face, for the line the page is arguing about.

    Drawn as positioned text with hairline rules and no verticals: ASCII tables and boxed
    grids are what make a PDF look like a terminal dump.
    """
    headers = [_clean(str(h)) for h in headers]
    body = [[_clean(str(v)) for v in r] for r in rows]
    ncol = len(headers)
    if aligns is None:
        aligns = ["right" if body and all(_is_num(r[k]) for r in body) and k > 0 else "left" for k in range(ncol)]
    aligns = list(aligns)

    def face(i: int, k: int) -> str:
        """The face a body cell is drawn in, which is the face it has to be measured in."""
        return HEADING_FONT if k == 0 or (hl_row is not None and i == hl_row) else BODY_FONT

    def widths(k: int) -> list:
        """Width of a column's header, then of each of its cells."""
        cells = [text_width(r[k], size, face(i, k)) for i, r in enumerate(body)]
        return [text_width(headers[k].upper(), 7.1, HEADING_FONT)] + cells

    # natural width of each column: widest of its header and its cells
    def natural(k: int) -> float:
        return max(widths(k))

    avail = x1 - x0 - gutter * (ncol - 1)
    nat = [natural(k) for k in range(ncol)]

    # Too wide for the column: set the body smaller, down to a floor that still reads
    while sum(nat) > avail and size > MIN_TABLE_SIZE:
        size = max(MIN_TABLE_SIZE, size - 0.3)
        nat = [natural(k) for k in range(ncol)]

    # Still too wide: trim the widest text of the widest column that can spare it, a few characters at a time
    for _ in range(TRIM_STEPS):
        if sum(nat) <= avail:
            break
        # A column is as wide as its widest text, so trimming only helps where that text can be cut
        for k in sorted(range(ncol), key=nat.__getitem__, reverse=True):
            column = widths(k)
            widest = max(range(len(column)), key=column.__getitem__)
            text = headers[k] if widest == 0 else body[widest - 1][k]
            if len(text) > MIN_CELL_CHARS:
                break
        else:
            break  # nothing left to trim
        trimmed = text.removesuffix("...")[:-3].rstrip() + "..."
        if widest == 0:
            headers[k] = trimmed
        else:
            body[widest - 1][k] = trimmed
        nat[k] = natural(k)

    slack = avail - sum(nat)
    if slack < 0:
        warnings.warn(
            f"table: content needs {sum(nat) + gutter * (ncol - 1):.3f} of {x1 - x0:.3f} "
            f"figure width even set small and trimmed; drop a column",
            stacklevel=2,
        )
        slack = 0.0
    step = gutter + (slack / (ncol - 1) if ncol > 1 else 0.0)

    # anchor per column: left edge for left-aligned, right edge for right-aligned
    lefts, cur = [], x0
    for k in range(ncol):
        lefts.append(cur)
        cur += nat[k] + step
    anchor = [lefts[k] if aligns[k] == "left" else lefts[k] + nat[k] for k in range(ncol)]

    for k, h in enumerate(headers):
        fig.text(
            anchor[k],
            y,
            h.upper(),
            fontsize=7.1,
            color=header_color,
            va="top",
            ha=aligns[k],
            fontname=HEADING_FONT,
            fontweight=HEADING_WEIGHT,
        )
    y -= 0.0145
    rule(fig, y + 0.004, x0=x0, x1=x1, c=RULE)
    y -= 0.0065

    for i, r in enumerate(body):
        hot = hl_row is not None and i == hl_row
        for k, v in enumerate(r):
            fig.text(
                anchor[k],
                y,
                v,
                fontsize=size,
                va="top",
                ha=aligns[k],
                color=INK if hot else BODY,
                fontname=face(i, k),
                fontweight=_weight(face(i, k)),
            )
        y -= lead
        if row_rules and i < len(body) - 1:
            rule(fig, y + 0.0048, x0=x0, x1=x1, c=HAIRLINE)
    return y - 0.008


def stat_row(
    fig: plt.Figure,
    x: float,
    y: float,
    pairs: Sequence[Tuple[str, str]],
    size: float = 8.6,
    dx: float = 0.082,
    gutter: float = 0.012,
) -> float:
    """Row of label-over-value stats. The cheapest way to put numbers near prose.

    `dx` is a **minimum** column pitch: the row measures its own labels and values and widens to
    fit, so a long label pushes the row out instead of overprinting its neighbour. Eyeballed
    pitches are the single most common source of overlap in a report.
    """
    needed = (
        max(
            [text_width(_clean(str(k)).upper(), 6.6, HEADING_FONT) for k, _ in pairs]
            + [text_width(str(v), size, HEADING_FONT) for _, v in pairs]
        )
        + gutter
    )
    dx = max(dx, needed)
    for i, (k, v) in enumerate(pairs):
        fig.text(
            x + i * dx,
            y,
            _clean(k).upper(),
            fontsize=6.6,
            color=MUTED,
            va="top",
            fontname=HEADING_FONT,
            fontweight=HEADING_WEIGHT,
        )
        fig.text(
            x + i * dx,
            y - 0.0125,
            _clean(str(v)),
            fontsize=size,
            color=INK,
            va="top",
            fontname=HEADING_FONT,
            fontweight=HEADING_WEIGHT,
        )
    return y - 0.030


def stat_tiles(
    fig: plt.Figure,
    y: float,
    pairs: Sequence[Tuple[str, str]],
    x0: float = L,
    x1: float = R,
    height: float = 0.05,
    gap: float = 0.01,
) -> float:
    """A page's headline numbers as a row of equal tiles: a large value over its label.

    A value too wide for its tile is set smaller, so a row never overprints. Returns the `y`
    to continue at.

    Args:
        fig (plt.Figure): The page.
        y (float): Top of the row.
        pairs (Sequence[Tuple[str, str]]): Label and value for each tile.
        x0 (float): Left edge of the row.
        x1 (float): Right edge of the row.
        height (float): Height of a tile.
        gap (float): Space between tiles.
    """
    width = (x1 - x0 - gap * (len(pairs) - 1)) / len(pairs)
    inset = 0.012
    for i, (label, value) in enumerate(pairs):
        left = x0 + i * (width + gap)
        panel(fig, left, y - height, left + width, y)
        label, value, size = _clean(str(label)).upper(), _clean(str(value)), 14.0
        while text_width(value, size, HEADING_FONT) > width - 2 * inset and size > 8:
            size -= 0.5
        fig.text(
            left + inset,
            y - 0.0085,
            value,
            fontsize=size,
            color=INK,
            va="top",
            fontname=HEADING_FONT,
            fontweight=HEADING_WEIGHT,
        )
        label_size = 6.4
        while text_width(label, label_size, HEADING_FONT) > width - 2 * inset and label_size > 5:
            label_size -= 0.2
        fig.text(
            left + inset,
            y - height + 0.008,
            label,
            fontsize=label_size,
            color=MUTED,
            fontname=HEADING_FONT,
            fontweight=HEADING_WEIGHT,
        )
    return y - height - 0.022


def footer(fig: plt.Figure, left: str, page: Optional[object] = None) -> None:
    """Quiet page furniture. Cheap, and it makes a loose printed page identifiable."""
    fig.text(L, 0.028, _clean(left), fontsize=7.6, color=MUTED)
    if page is not None:
        fig.text(R, 0.028, str(page), fontsize=7.6, color=MUTED, ha="right")


def tidy(ax, title: str = "", grid: str = "both") -> None:
    """Report styling for an axes: no top/right spines, hairline grid, left-aligned title.

    Args:
        ax: Axes to style.
        title (str): Left-aligned title, set in the heading font.
        grid (str): Which gridlines to draw: "both", "x" or "y".
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, labelsize=8)
    if title:
        ax.set_title(
            title, loc="left", pad=7, fontsize=9.5, fontname=HEADING_FONT, fontweight=HEADING_WEIGHT, color=INK
        )


def _titles(ax) -> List:
    """Every title artist on an axes.

    `set_title(loc="left")` does not write to `ax.title` -- matplotlib keeps separate
    `_left_title` and `_right_title` artists, and `ax.title` stays empty. Checking only
    `ax.title` therefore misses every left-aligned title, which is the house style here.
    """
    found = [getattr(ax, name, None) for name in ("title", "_left_title", "_right_title")]
    return [t for t in found if t is not None and str(t.get_text()).strip()]


def _on_axes_ticks(ax, renderer) -> List:
    """Tick labels that are really drawn.

    matplotlib keeps tick artists for ticks outside the view at stale positions, so an
    unfiltered sweep reports a `2.4` or a `10^25` that no reader will ever see. On this
    report that noise was 18 of 23 hits.
    """
    box = ax.get_window_extent(renderer).padded(30)
    return [
        t
        for t in ax.get_xticklabels() + ax.get_yticklabels()
        if t.get_visible() and box.overlaps(t.get_window_extent(renderer))
    ]


def overlaps(fig: plt.Figure, min_frac: float = 0.12, pad: float = -0.6) -> List[Dict]:
    """Pairs of text on `fig` whose boxes genuinely collide.

    Compared by **overlap fraction** of the smaller box, not raw intersection: consecutive
    lines of a paragraph always abut a little, so an intersection test flags every
    paragraph ever written. `pad` shrinks each box to discount that leading further.

    Each hit gives both strings and the overlap in points, which is usually enough to see
    which column needs moving.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    artists = list(fig.texts)
    for ax in fig.axes:
        artists += list(ax.texts) + _titles(ax) + [ax.xaxis.label, ax.yaxis.label]
        artists += _on_axes_ticks(ax, renderer)
        if ax.get_legend():
            artists += list(ax.get_legend().get_texts())
    boxes = []
    for t in artists:
        if not t.get_visible() or not str(t.get_text()).strip():
            continue
        try:
            box = t.get_window_extent(renderer=renderer).padded(pad)
        except Exception:
            continue
        if box.width > 0 and box.height > 0:
            boxes.append((t, box))
    hits = []
    for i, (ta, ba) in enumerate(boxes):
        for tb, bb in boxes[i + 1 :]:
            ix = max(0.0, min(ba.x1, bb.x1) - max(ba.x0, bb.x0))
            iy = max(0.0, min(ba.y1, bb.y1) - max(ba.y0, bb.y0))
            if ix <= 0 or iy <= 0:
                continue
            frac = ix * iy / min(ba.width * ba.height, bb.width * bb.height)
            if frac >= min_frac:
                hits.append(
                    {
                        "a": str(ta.get_text())[:40],
                        "b": str(tb.get_text())[:40],
                        "frac": round(float(frac), 2),
                        "dx_pt": round(ix / fig.dpi * 72, 1),
                        "dy_pt": round(iy / fig.dpi * 72, 1),
                    }
                )
    return hits


class _OverlapAudit:
    """Stands in for a PdfPages: audits each page instead of writing it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pages: List[List[Dict]] = []

    def savefig(self, fig, **_):
        self.pages.append(overlaps(fig, **self.kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def audit(build: Callable, *args, min_frac: float = 0.12, pad: float = -0.6, **kwargs) -> Dict[str, List]:
    """Lint a report without writing it: text collisions and missing glyphs.

    `build` is called with an object that quacks like a `PdfPages`, so pages are measured
    instead of saved and nothing is overwritten while you iterate. Structure a report as a
    `build(pdf)` plus a thin saver that opens the real one, and it stays lintable.

    Both failure modes are otherwise silent: matplotlib substitutes a missing glyph with a
    visible box without raising, and there is no layout engine for free figure text --
    `tight_layout` and `constrained_layout` only manage axes.
    """
    shim = _OverlapAudit(min_frac=min_frac, pad=pad)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build(shim, *args, **kwargs)
    return {
        "overlaps": [dict(page=n, **hit) for n, hits in enumerate(shim.pages, 1) for hit in hits],
        "glyphs": sorted({str(w.message) for w in caught if "Glyph" in str(w.message)}),
    }


def fit_axes(fig: plt.Figure, pad: float = 9.0) -> List[Dict]:
    """Pull each axes in until its real extent clears the text around it, by `pad` points.

    An axes draws its title above its rect, and tick labels and an axis label below it --
    all outside the rect you handed `add_axes`. Figure text placed with a hand-run cursor
    knows nothing about that band, so the two collide: a left title striking the paragraph
    above, an x-axis label striking the caption below. `ax.get_tightbbox` is the authority
    on what an axes really occupies, decorations included.

    Measures each axes against the figure text and other axes sharing its x-range, then
    shrinks the rect until the real extent clears them. Call it on each page just before
    saving; returns what it moved, for the record.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    to_pt = 72.0 / fig.dpi
    pad_px = pad / to_pt
    text_boxes = [t.get_window_extent(renderer) for t in fig.texts if t.get_visible() and str(t.get_text()).strip()]
    tight = {ax: ax.get_tightbbox(renderer) for ax in fig.axes}
    moved = []
    for ax in fig.axes:
        own, box = tight[ax], ax.get_window_extent(renderer)
        if own is None:
            continue
        blockers = text_boxes + [b for a, b in tight.items() if a is not ax and b is not None]
        mid = 0.5 * (own.y0 + own.y1)
        near = [b for b in blockers if b.x1 > own.x0 and b.x0 < own.x1]
        above = [b for b in near if b.y0 > mid]
        below = [b for b in near if b.y1 < mid]
        top = max(0.0, own.y1 - (min(b.y0 for b in above) - pad_px)) if above else 0.0
        bot = max(0.0, (max(b.y1 for b in below) + pad_px) - own.y0) if below else 0.0
        if top < 0.5 and bot < 0.5:
            continue
        height = box.height - bot - top
        if height <= 0:
            warnings.warn(f"fit_axes: no room left for axes {ax.get_label() or ax!r}", stacklevel=2)
            continue
        inv = fig.transFigure.inverted()
        (x0, y0), (x1, y1) = inv.transform([(box.x0, box.y0 + bot), (box.x1, box.y0 + bot + height)])
        ax.set_position([x0, y0, x1 - x0, y1 - y0])
        moved.append(
            {
                "axes": ax.get_label() or f"axes@{round(box.y0 * to_pt)}pt",
                "shrink_top_pt": round(top * to_pt, 1),
                "shrink_bottom_pt": round(bot * to_pt, 1),
            }
        )
    return moved


def save(path: Union[str, Path], build: Callable, *args, overwrite: bool = False, **kwargs) -> Dict[str, List]:
    """Lint a report with `audit`, then write it.

    Args:
        path (Union[str, Path]): Where to write the PDF. Parent directories are created.
        build (Callable): Called as `build(pdf, *args, **kwargs)`; saves each page with
            `pdf.savefig(fig)`.
        overwrite (bool): Replace an existing file. Defaults to False.

    Returns:
        Dict[str, List]: The `audit` result. Each problem is also logged as a warning.

    Raises:
        FileExistsError: If `path` exists and `overwrite` is False.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    lint = check(build, *args, **kwargs)
    write(path, build, *args, **kwargs)
    return lint


def check(build: Callable, *args, **kwargs) -> Dict[str, List]:
    """Apply the theme, build every page without writing, and log each lint problem.

    Returns:
        Dict[str, List]: The `audit` result.
    """
    theme(HEADING_FONT, BODY_FONT)
    lint = audit(build, *args, **kwargs)
    for problem in lint["overlaps"] + lint["glyphs"]:
        log.warning(f"Report lint: {problem}")
    return lint


def write(path: Union[str, Path], build: Callable, *args, **kwargs) -> None:
    """Write a report's pages to a PDF, replacing any file at `path`.

    Args:
        path (Union[str, Path]): Where to write the PDF. Parent directories are created.
        build (Callable): Called as `build(pdf, *args, **kwargs)`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        build(pdf, *args, **kwargs)
    log.info(f"Wrote {path.resolve()} ({path.stat().st_size / 1024:.0f} KB)")
