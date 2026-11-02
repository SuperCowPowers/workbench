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
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg

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


@lru_cache(maxsize=4096)
def _measured(text: str, size: float, fontname: str, figw: float) -> float:
    """Rendered width of a string in figure-fraction units. Cached -- it is a real render."""
    fig = plt.figure(figsize=(figw, 11))
    canvas = FigureCanvasAgg(fig)
    t = fig.text(0, 0, text, fontsize=size, fontname=fontname)
    width = t.get_window_extent(renderer=canvas.get_renderer()).width / (fig.dpi * figw)
    plt.close(fig)
    return width


def text_width(text: str, size: float = 9.5, font: Optional[str] = None,
               figw: float = 8.5) -> float:
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


@lru_cache(maxsize=2048)
def _measured_height(text: str, size: float, fontname: str, figh: float) -> float:
    """Rendered height of a string in figure-fraction units, descenders included."""
    fig = plt.figure(figsize=(8.5, figh))
    canvas = FigureCanvasAgg(fig)
    t = fig.text(0, 0.5, text, fontsize=size, fontname=fontname)
    height = t.get_window_extent(renderer=canvas.get_renderer()).height / (fig.dpi * figh)
    plt.close(fig)
    return height


def text_height(text: str, size: float = 9.5, font: Optional[str] = None, figh: float = 11.0) -> float:
    """How tall a string will actually be, in figure-fraction units.

    Stack things with this rather than a constant: a title with descenders is taller than one
    without, so a fixed gap under it collides for some titles and gaps for others.
    """
    return _measured_height(text, size, font or BODY_FONT, figh)


def head(fig: plt.Figure, kicker: str, title: str, sub: Optional[str] = None) -> float:
    """Page header: letterspaced kicker, large title, optional standfirst, rule.

    The standfirst sits under the title's *measured* bottom, so a title with descenders does
    not crowd it and one without does not leave a hole. Returns the `y` to start the body at.
    """
    title, size = _clean(title), 25.0
    fig.text(L, 0.962, "  ".join(_clean(kicker).upper()), fontname=HEADING_FONT, fontsize=7.6, color=MUTED)
    fig.text(L, 0.936, title, fontname=HEADING_FONT, fontsize=size, color=INK, va="top")
    y = 0.936 - _measured_height(title, size, HEADING_FONT, 11.0) - 0.007
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
        aligns = [
            "right" if body and all(_is_num(r[k]) for r in body) and k > 0 else "left"
            for k in range(ncol)
        ]
    aligns = list(aligns)

    # natural width of each column: widest of its header and its cells
    nat = []
    for k in range(ncol):
        cells = [text_width(r[k], size, HEADING_FONT) for r in body]
        nat.append(max([text_width(headers[k].upper(), 6.9, HEADING_FONT)] + cells))

    avail = x1 - x0
    slack = avail - (sum(nat) + gutter * (ncol - 1))
    if slack < 0:
        warnings.warn(
            f"table: content needs {sum(nat) + gutter * (ncol - 1):.3f} of {avail:.3f} "
            f"figure width; shorten labels or drop a column",
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
        fig.text(anchor[k], y, h.upper(), fontsize=7.1, color=header_color, va="top",
                 ha=aligns[k], fontname=HEADING_FONT)
    y -= 0.0145
    rule(fig, y + 0.004, c=RULE)
    y -= 0.0065

    for i, r in enumerate(body):
        hot = hl_row is not None and i == hl_row
        for k, v in enumerate(r):
            fig.text(anchor[k], y, v, fontsize=size, va="top", ha=aligns[k],
                     color=INK if hot else BODY,
                     fontname=HEADING_FONT if hot or k == 0 else BODY_FONT)
        y -= lead
        if row_rules and i < len(body) - 1:
            rule(fig, y + 0.0048, c=HAIRLINE)
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
            [text_width(str(k), 6.6, HEADING_FONT) for k, _ in pairs]
            + [text_width(str(v), size, HEADING_FONT) for _, v in pairs]
        )
        + gutter
    )
    dx = max(dx, needed)
    for i, (k, v) in enumerate(pairs):
        fig.text(x + i * dx, y, _clean(k).upper(), fontsize=6.6, color=MUTED, va="top", fontname=HEADING_FONT)
        fig.text(x + i * dx, y - 0.0125, _clean(str(v)), fontsize=size, color=INK, va="top", fontname=HEADING_FONT)
    return y - 0.030


def footer(fig: plt.Figure, left: str, page: Optional[object] = None) -> None:
    """Quiet page furniture. Cheap, and it makes a loose printed page identifiable."""
    fig.text(L, 0.028, _clean(left), fontsize=7.6, color=MUTED)
    if page is not None:
        fig.text(R, 0.028, str(page), fontsize=7.6, color=MUTED, ha="right")


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
    return [t for t in ax.get_xticklabels() + ax.get_yticklabels()
            if t.get_visible() and box.overlaps(t.get_window_extent(renderer))]


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
        for tb, bb in boxes[i + 1:]:
            ix = max(0.0, min(ba.x1, bb.x1) - max(ba.x0, bb.x0))
            iy = max(0.0, min(ba.y1, bb.y1) - max(ba.y0, bb.y0))
            if ix <= 0 or iy <= 0:
                continue
            frac = ix * iy / min(ba.width * ba.height, bb.width * bb.height)
            if frac >= min_frac:
                hits.append({
                    "a": str(ta.get_text())[:40],
                    "b": str(tb.get_text())[:40],
                    "frac": round(float(frac), 2),
                    "dx_pt": round(ix / fig.dpi * 72, 1),
                    "dy_pt": round(iy / fig.dpi * 72, 1),
                })
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
    text_boxes = [
        t.get_window_extent(renderer)
        for t in fig.texts
        if t.get_visible() and str(t.get_text()).strip()
    ]
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
        moved.append({
            "axes": ax.get_label() or f"axes@{round(box.y0 * to_pt)}pt",
            "shrink_top_pt": round(top * to_pt, 1),
            "shrink_bottom_pt": round(bot * to_pt, 1),
        })
    return moved




