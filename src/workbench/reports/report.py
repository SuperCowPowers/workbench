"""Report: a PDF document built from sections, with the rows behind it in the DataFrame Store"""

import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

from workbench.reports.paginator import TABLE_HEAD, TABLE_ROW, Paginator
from workbench.utils.plots import report as pages

log = logging.getLogger("workbench")

# Summary page layout, in figure units: the height of a card and the gap under it
CARD_HEIGHT, CARD_GAP = 0.2, 0.014

# Inside a card: the inset of its content, where its chart starts, and the characters of a headline line
CARD_PAD, CARD_CHART_X, CARD_TEXT_WIDTH = 0.02, 0.62, 66

# A card's status: its wording and the colour of its dot
STATUS = {True: ("Needs a look", pages.ACCENTS[2]), False: ("Fine", pages.ACCENTS[0])}

# A note's severity, most severe first: the colour of the badge that carries its number
SEVERITY = {"serious": "#B42318", "warning": "#A16207", "info": "#15803D"}

# A printed note: the characters on a line, the pitch between lines, and the most rows of its details shown
NOTE_WIDTH, NOTE_LEAD, NOTE_DETAIL_ROWS = 104, 0.0165, 10


def _thumbnail(ax) -> None:
    """Strip a chart down to read at card size: no title, axis labels or legend, small ticks."""
    ax.set_title("")
    ax.set_title("", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.tick_params(labelsize=6.5)


@dataclass
class Summary:
    """A section's card on the report's summary page.

    Attributes:
        label (str): What the section is, e.g. "Model performance".
        headline (str): The finding, in one sentence.
        stats (List[Tuple[str, str]]): Up to four label/value pairs.
        chart (Optional[Callable]): Called with an axes to draw the section's lead chart, small.
        hero (Optional[Tuple[str, str]]): The one number the card leads with, and what it counts.
        needs_action (Optional[bool]): Whether the section found something to act on. None
            leaves the card without a status.
    """

    label: str
    headline: str
    stats: List[Tuple[str, str]] = field(default_factory=list)
    chart: Optional[Callable] = None
    hero: Optional[Tuple[str, str]] = None
    needs_action: Optional[bool] = None


@dataclass(frozen=True, eq=False)
class Note:
    """Something a reader must know that the report's sections cannot work out for themselves.

    A note about an artifact is printed once, in the section about that artifact, and every other
    section that involves the artifact carries a numbered marker that links to it. A note about
    nothing in particular is printed on the summary page.

    Attributes:
        text (str): The note.
        severity (str): "serious" (a red badge), "warning" (dark yellow) or "info" (green). A
            serious note marks its sections' cards as needing a look.
        about (Optional[str]): The artifact it concerns, by the name the sections use for it
            (an Archive, a DataSource, a FeatureSet). None for a general note.
        details (Optional[pd.DataFrame]): The rows the note is about. The first few are tabled
            under the note, and the whole frame is published with the report's other frames.
    """

    text: str
    severity: str = "info"
    about: Optional[str] = None
    details: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if self.severity not in SEVERITY:
            raise ValueError(f"severity is one of {list(SEVERITY)}, not {self.severity!r}")


@dataclass
class Section:
    """Pages for a report, the detail frames behind them, and a card for the summary page.

    Attributes:
        pages (List[Callable]): Each is called with the Report and returns one page figure, or
            a list of them. Lay a section out with a `Paginator` and return its `figs`; the
            Report adds the footer and page number, fits the axes and saves each page.
        frames (Dict[str, pd.DataFrame]): Detail frames by name, published to the DataFrame
            Store when the report is saved. A page gets a frame's location from
            `report.location(name)`.
        summary (Optional[Summary]): This section's card on the summary page.
        about (List[str]): The artifacts this section is about, by name. The report's notes on
            those artifacts are shown where the section calls `Paginator.notes`.
    """

    pages: List[Callable] = field(default_factory=list)
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: Optional[Summary] = None
    about: List[str] = field(default_factory=list)


class Report:
    """Report: a PDF document built from sections, with the rows behind it in the DataFrame Store.

    Sections come from `workbench.reports.sections` (model performance, data quality, data
    comparison) or are your own page functions. Page 1 is always a summary: the author's
    sentence, what the report is limited to, then one card per section
    with its finding, key numbers and lead chart. Clicking a card opens its section's first
    page, every later page links back to the summary, and the sections are the PDF's bookmarks.
    A `Note` about an artifact is printed in that artifact's section and linked from the others. Saving
    lints every page, writes the PDF, and publishes each section's detail frames to
    `/reports/<name>/<frame>` so a reader can go from a count in the document to the rows
    behind it.

    Common Usage:
        ```python
        from workbench.reports import Report, sections
        from workbench.utils.pandas_utils import compare_keyed, data_quality

        report = Report("Mouse PK intake", summary="The collected arms match the DataSource.")
        report.add(sections.data_quality(data_quality(collected, key="batch")))
        report.add(sections.comparison(compare_keyed(collected, existing, on="batch"), "Archive", "DataSource"))
        report.save("mouse_pk_intake.pdf")
        ```
    """

    def __init__(
        self,
        title: str,
        name: Optional[str] = None,
        summary: Optional[str] = None,
        filters: Optional[List[str]] = None,
        notes: Optional[List[Note]] = None,
    ):
        """Report Init

        Args:
            title (str): Shown above every page headline and in the footer.
            name (Optional[str]): Names the report's DataFrame Store subtree, /reports/<name>.
                Defaults to the title in lowercase with underscores.
            summary (Optional[str]): The author's conclusion, shown at the top of the summary page.
            filters (Optional[List[str]]): What the report is limited to, one line each (species,
                route, date range, which rows were left out and why). Listed on the summary
                page so a reader knows what the numbers cover.
            notes (Optional[List[Note]]): What a reader must know that the sections cannot work
                out. Numbered in this order.
        """
        self.filters = list(filters or [])
        self.notes: List[Note] = list(notes or [])
        self._abouts: List[List[str]] = []  # what each section added is about
        self._homes: Dict[int, plt.Figure] = {}  # the page each note is printed on, by note index
        self._marks: List[Tuple[plt.Figure, Tuple[float, float, float, float], int, object]] = []
        self.title = title
        self.summary = summary
        # Each card with the index of its section's first page function and what the section is about
        self._cards: List[Tuple[Summary, int, List[str]]] = []
        self._links: List[Tuple[int, Tuple[float, float, float, float], int]] = []
        self._bookmarks: List[Tuple[str, int]] = []
        self.name = name or re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        self.frames: Dict[str, pd.DataFrame] = {
            f"note_{number}": note.details
            for number, note in enumerate(self.notes, start=1)
            if note.details is not None
        }
        self._pages: List[Callable] = []
        self._published = False

    def add(self, section: Union[Section, Callable]) -> "Report":
        """Add a section, or a single page function, to the end of the report.

        Args:
            section (Union[Section, Callable]): A Section, or a function taking this Report
                and returning one page figure.

        Returns:
            Report: This report, so calls chain.

        Raises:
            ValueError: If the section carries a frame name the report already has.
        """
        if not isinstance(section, Section):
            section = Section(pages=[section])
        repeated = sorted(set(section.frames) & set(self.frames))
        if repeated:
            raise ValueError(f"Frames {repeated} are already in the report; give the section its own name")
        self._abouts.append(list(section.about))
        if section.summary is not None:
            self._cards.append((section.summary, len(self._pages), list(section.about)))
        self._pages.extend(section.pages)
        self.frames.update(section.frames)
        return self

    def new_page(self, headline: str, sub: Optional[str] = None, continued: bool = False) -> Tuple[plt.Figure, float]:
        """Start a page: the report title as the kicker, then the headline and standfirst.

        Args:
            headline (str): The page's one idea.
            sub (Optional[str]): A sentence under the headline.
            continued (bool): Mark the page as carrying on from the one before it.

        Returns:
            Tuple[plt.Figure, float]: The page figure and the `y` to start the body at.
        """
        fig = pages.new_page()
        return fig, pages.head(fig, self.title, headline, sub, tag="Continued" if continued else None)

    def location(self, frame: str) -> Optional[str]:
        """Where a detail frame is published, or None when the report is saved without publishing.

        Args:
            frame (str): The frame's name in its section.

        Returns:
            Optional[str]: The DataFrame Store location, for `DFStore().get(location)`.
        """
        return f"/reports/{self.name}/{frame}" if self._published else None

    def _home(self, note: Note) -> Optional[Tuple[str, ...]]:
        """What the section a note is printed in is about: its artifact alone, else the first to involve it."""
        involved = [tuple(about) for about in self._abouts if note.about in about]
        alone = [about for about in involved if len(about) == 1]
        return (alone or involved or [None])[0]

    def note_blocks(self, about: List[str]) -> List[Tuple[float, Callable]]:
        """The report's notes on some artifacts, as blocks for a section's page to place.

        A note is printed in full in the section it belongs with, under a badge carrying its
        number in its severity's colour. Any other section gets one line: the badge and the page
        that has the note in full, which it links to.

        Args:
            about (List[str]): The artifacts the section is about.

        Returns:
            List[Tuple[float, Callable]]: For each note, the height it needs and a function
                `draw(fig, y)` that draws it and returns the `y` to continue at.
        """

        def printed(index: int) -> Callable:
            def draw(fig, y):
                self._homes[index] = fig
                return self._print_note(fig, y, index)

            return draw

        def marker(index: int) -> Callable:
            def draw(fig, y):  # the line's wording is filled in once the page that has the note is numbered
                note = self.notes[index]
                right = pages.badge(fig, pages.L, y, f"Note [{index + 1}]", SEVERITY[note.severity])
                line = fig.text(right + 0.01, y - 0.0025, "", fontsize=8.8, color=pages.ACCENTS[1], va="top")
                self._marks.append((fig, (pages.L, y - 0.019, pages.R, y + 0.003), index, line))
                return y - 0.03

            return draw

        blocks, claimed = [], set()
        for index, note in enumerate(self.notes):
            if note.about is None or note.about not in about:
                continue
            if tuple(about) == self._home(note) and index not in self._homes and index not in claimed:
                claimed.add(index)
                blocks.append((self._note_height(index) + 0.018, printed(index)))
            else:
                blocks.append((0.03, marker(index)))
        return blocks

    def _note_height(self, index: int) -> float:
        """The height of a note's panel: its badge line, its text, and its details table when it has one."""
        note = self.notes[index]
        height = 0.05 + len(textwrap.wrap(note.text, NOTE_WIDTH)) * NOTE_LEAD
        if note.details is not None and len(note.details):
            height += TABLE_HEAD + min(len(note.details), NOTE_DETAIL_ROWS) * TABLE_ROW + 0.03
        return height

    def _print_note(self, fig: plt.Figure, y: float, index: int) -> float:
        """One note in full, in a panel under its severity badge, with the first rows of its details."""
        note = self.notes[index]
        lines = textwrap.wrap(note.text, NOTE_WIDTH)
        height, x = self._note_height(index), pages.L + 0.014
        pages.panel(fig, pages.L, y - height, pages.R, y)
        right = pages.badge(fig, x, y - 0.012, f"Note [{index + 1}]", SEVERITY[note.severity])
        label = f"on {note.about}" if note.about else ""
        fig.text(right + 0.01, y - 0.0142, label, fontsize=8.6, color=pages.BODY, va="top").set(
            fontname=pages.HEADING_FONT, fontweight=pages.HEADING_WEIGHT
        )
        text_size = {"size": 9.6, "width": NOTE_WIDTH, "color": pages.INK, "x": x, "lead": NOTE_LEAD}
        below = pages.para(fig, y - 0.036, "\n".join(lines), **text_size)
        if note.details is not None and len(note.details):
            shown = note.details.head(NOTE_DETAIL_ROWS)
            cells = [
                ["" if pd.isna(v) else f"{v:g}" if isinstance(v, float) else str(v) for v in row]
                for row in shown.to_numpy()
            ]
            below = pages.table(fig, below - 0.008, list(shown.columns), cells, x0=x, x1=pages.R - 0.014)
            location, count = self.location(f"note_{index + 1}"), len(note.details)
            where = f"Showing the first {len(shown):,} of {count:,}. " if len(shown) < count else ""
            where += f"All {count:,} rows: DFStore {location}" if location else ""
            fig.text(x, below + 0.002, where, fontsize=8.2, color=pages.MUTED, va="top")
        return y - height - 0.018

    def build(self, pdf) -> None:
        """Draw every page onto a PdfPages-like object, numbering them in order.

        Args:
            pdf: An object with `savefig(fig)`.
        """
        if not self.summary and not self._cards:
            raise ValueError("A report needs a summary: pass summary= or add a section that has one")
        adrift = sorted({note.about for note in self.notes if note.about and self._home(note) is None})
        if adrift:
            raise ValueError(f"Notes are about {adrift}, which no section is about; pass about= to a section")
        self._homes, self._marks = {}, []
        published = {name: df for name, df in self.frames.items() if self._published and not df.empty}
        # The summary is drawn last, once the body pages are counted, and saved first
        with plt.rc_context({"figure.max_open_warning": 0}):
            body, starts = [], []
            for page in self._pages + ([self._data_page] if published else []):
                starts.append(len(body))
                drawn = page(self)
                body += drawn if isinstance(drawn, list) else [drawn]
            summary = self._summary_pages(starts)
            number_of = {id(fig): number for number, fig in enumerate(summary + body, start=1)}
            for fig, box, index, line in self._marks:  # each marker names and opens the page that has its note
                note, home = self.notes[index], number_of[id(self._homes[index])]
                line.set_text(f"on {note.about}: see page {home}")
                self._links.append((number_of[id(fig)] - 1, box, home - 1))
            for number, fig in enumerate(summary + body, start=1):
                if number > len(summary):
                    self._links.append((number - 1, self._back_link(fig), 0))
                pages.footer(fig, self.title, number)
                pages.fit_axes(fig)
                pdf.savefig(fig)
                plt.close(fig)

    def save(self, path: Union[str, Path], overwrite: bool = False, publish: bool = True) -> Dict[str, List]:
        """Lint the pages, publish the detail frames, and write the PDF.

        Every page is built before anything is replaced, so a page that raises leaves the
        previous run's PDF and published frames as they were.

        Args:
            path (Union[str, Path]): Where to write the PDF.
            overwrite (bool): Replace an existing PDF. Defaults to False.
            publish (bool): Publish the detail frames to /reports/<name>/ in the DataFrame
                Store, replacing the previous run's. Defaults to True.

        Returns:
            Dict[str, List]: The lint result from `plots.report.audit`.

        Raises:
            FileExistsError: If `path` exists and `overwrite` is False. Nothing is published.
        """
        if Path(path).exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        self._published = publish
        lint = pages.check(self.build)
        if publish:
            self._publish_frames()
        pages.write(path, self.build)
        pages.navigate(path, self._links, self._bookmarks, self.title)
        return lint

    def _publish_frames(self) -> None:
        """Upsert each frame under /reports/<name>/; an empty frame removes the previous run's."""
        from workbench.api.df_store import DFStore

        store = DFStore(path_prefix=f"/reports/{self.name}")
        for name, df in self.frames.items():
            if not df.empty:
                store.upsert(f"/{name}", df)
            elif store.check(f"/{name}"):
                store.delete(f"/{name}")

    def _summary_pages(self, starts: List[int]) -> List[plt.Figure]:
        """The summary: the author's sentence, the filters, the general notes, then a card per section.

        Args:
            starts (List[int]): For each page function, the body page its first figure is.
        """
        flow = Paginator(self, "Summary", self.summary)
        if self.filters:
            flow.label("What this report covers", pages.MUTED)
            flow.lines([f"-  {line}" for line in self.filters])
            flow.block(0.028, lambda fig, y: pages.rule(fig, y - 0.004, c=pages.HAIRLINE))
        for index, note in enumerate(self.notes):
            if note.about is None:

                def general(fig, y, index=index):
                    self._homes[index] = fig
                    return self._print_note(fig, y, index)

                flow.block(self._note_height(index) + 0.018, general)
        placed = []  # each card's page text and box, numbered once the summary's own page count is known
        for card, first, about in self._cards:

            def draw(fig, y, card=card, first=first, about=about):
                box = (pages.L, y - CARD_HEIGHT, pages.R, y)
                text = self._card(fig, card, box, self._worst_note(about))
                placed.append((text, len(flow.figs) - 1, box, starts[first]))

            flow.block(CARD_HEIGHT + CARD_GAP, draw)
        figs = flow.figs
        self._links = [(on, box, len(figs) + target) for _, on, box, target in placed]
        self._bookmarks = [("Summary", 0)]
        for (card, _, _), (text, _, _, target) in zip(self._cards, placed):
            text.set_text(f"PAGE {len(figs) + target + 1}")
            self._bookmarks.append((card.label, len(figs) + target))
        if len(starts) > len(self._pages):
            self._bookmarks.append(("Data behind this report", len(figs) + starts[-1]))
        return figs

    @staticmethod
    def _back_link(fig: plt.Figure) -> Tuple[float, float, float, float]:
        """Draw the way back to the summary in a page's top right corner; returns its box."""
        y, size, heading = 0.962, 7.6, pages.HEADING_FONT
        fig.text(
            pages.R,
            y,
            "SUMMARY",
            fontname=heading,
            fontweight=pages.HEADING_WEIGHT,
            fontsize=size,
            color=pages.ACCENTS[1],
            ha="right",
        )
        width, height = pages.text_width("SUMMARY", size, heading), pages.text_height("SUMMARY", size, heading)
        return (pages.R - width - 0.01, y - 0.006, pages.R + 0.01, y + height + 0.006)

    def _worst_note(self, about: List[str]) -> Optional[int]:
        """The most severe note on any of some artifacts, by index; None when there is none."""
        on = [index for index, note in enumerate(self.notes) if note.about in about]
        return min(on, key=lambda index: list(SEVERITY).index(self.notes[index].severity), default=None)

    def _card(self, fig: plt.Figure, card: Summary, box: Tuple[float, float, float, float], noted: Optional[int]):
        """Draw one section's card in `box`; returns the text that names the page it opens.

        `noted` is the most severe note on what the section is about, shown as a badge beside
        the status; a serious one makes the status "Needs a look".
        """
        note = self.notes[noted] if noted is not None else None
        needs_action = True if note is not None and note.severity == "serious" else card.needs_action
        x0, y0, x1, top = box
        pages.panel(fig, x0, y0, x1, top)
        x, y, heading = x0 + CARD_PAD, top - 0.018, pages.HEADING_FONT
        label = fig.text(
            x,
            y,
            card.label.upper(),
            fontname=heading,
            fontweight=pages.HEADING_WEIGHT,
            fontsize=8.4,
            color=pages.BODY,
            va="top",
        )
        after = x + pages.text_width(label.get_text(), 8.4, heading) + 0.016
        if needs_action is not None:
            wording, color = STATUS[needs_action]
            fig.text(after, y, "\u25cf", fontsize=7.5, color=color, va="top", fontname="DejaVu Sans")
            fig.text(after + 0.014, y, wording, fontsize=8.2, color=pages.BODY, va="top")
            after += 0.014 + pages.text_width(wording, 8.2) + 0.014
        if note is not None:
            pages.badge(fig, after, y + 0.0025, f"Note [{noted + 1}]", SEVERITY[note.severity])
        page = fig.text(
            x1 - CARD_PAD,
            y,
            "",
            fontname=heading,
            fontweight=pages.HEADING_WEIGHT,
            fontsize=7.6,
            color=pages.ACCENTS[1],
            va="top",
            ha="right",
        )
        y -= 0.026
        if card.hero is not None:
            value, counts = card.hero
            fig.text(
                x, y, value, fontname=heading, fontweight=pages.HEADING_WEIGHT, fontsize=25, color=pages.INK, va="top"
            )
            caption_x = x + pages.text_width(value, 25, heading) + 0.012
            fig.text(caption_x, y - 0.017, counts, fontsize=9, color=pages.MUTED, va="top")
            y -= 0.046
        size, lead = (9.4, 0.0155) if card.hero is not None else (11.5, 0.019)
        chars = CARD_TEXT_WIDTH if card.hero is not None else 54
        pages.para(fig, y, card.headline, size=size, width=chars, color=pages.INK, x=x, lead=lead)
        if card.stats:
            pages.stat_row(fig, x, y0 + 0.046, [(k, str(v)) for k, v in card.stats[:4]], dx=0.11)
        if card.chart is not None:
            ax = fig.add_axes([CARD_CHART_X, y0 + 0.018, x1 - CARD_PAD - CARD_CHART_X, CARD_HEIGHT - 0.054])
            ax.set_facecolor("none")
            card.chart(ax)
            _thumbnail(ax)
        return page

    def _data_page(self, _report: "Report") -> List[plt.Figure]:
        """The closing pages: every published frame, its size, and where to get it."""
        flow = Paginator(
            self, "Data behind this report", "Every count in this report has its rows in the DataFrame Store."
        )
        rows = [[name, f"{len(df):,}", self.location(name)] for name, df in self.frames.items() if not df.empty]
        flow.table(["frame", "rows", "location"], rows, aligns=["left", "right", "left"])
        flow.gap(0.02)
        flow.label("Reading one", pages.MUTED)
        flow.para(f'from workbench.api import DFStore\ndf = DFStore().get("{rows[0][2]}")', size=8.8)
        return flow.figs
