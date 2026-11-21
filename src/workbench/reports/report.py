"""Report: a PDF document built from sections, with the rows behind it in the DataFrame Store"""

import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

from workbench.utils.plots import report as pages

log = logging.getLogger("workbench")

# Summary page layout, in figure units: the height of a card, the lowest the page body may reach,
# the pitch of a filter or note line, and the characters on one
CARD_HEIGHT, SUMMARY_BOTTOM, SUMMARY_LEAD, SUMMARY_WIDTH = 0.185, 0.07, 0.0155, 112

# Footnotes: the baseline of the lowest line, the pitch between lines, and characters per line.
# A page body stops at sections.BOTTOM, which leaves room for FOOTNOTE_LINES lines.
FOOTNOTE_BASE, FOOTNOTE_LEAD, FOOTNOTE_WIDTH, FOOTNOTE_LINES = 0.047, 0.0125, 128, 3
IN_FULL = " (in full on page 1)"


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
    """

    label: str
    headline: str
    stats: List[Tuple[str, str]] = field(default_factory=list)
    chart: Optional[Callable] = None


@dataclass
class Section:
    """Pages for a report, the detail frames behind them, and a card for the summary page.

    Attributes:
        pages (List[Callable]): Each is called with the Report and returns one page figure,
            or a list of them when its content runs over. Start a page with
            `report.new_page(headline)`; the Report adds the footer and page number, fits
            the axes and saves it.
        frames (Dict[str, pd.DataFrame]): Detail frames by name, published to the DataFrame
            Store when the report is saved. A page gets a frame's location from
            `report.location(name)`.
        summary (Optional[Summary]): This section's card on the summary page.
        notes (List[str]): Notes this section refers to. The Report numbers them; a page marks
            one with `report.mark(note)` and prints it at its foot with `report.footnote(fig, note)`.
    """

    pages: List[Callable] = field(default_factory=list)
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: Optional[Summary] = None
    notes: List[str] = field(default_factory=list)


class Report:
    """Report: a PDF document built from sections, with the rows behind it in the DataFrame Store.

    Sections come from `workbench.reports.sections` (model performance, data quality, data
    comparison) or are your own page functions. Page 1 is always a summary: the author's
    sentence, what the report is limited to, its numbered notes, then one card per section
    with its finding, key numbers and lead chart. A note is marked where it applies and
    printed again at the foot of that page. Saving
    lints every page, writes the PDF, and publishes each section's detail frames to
    `/reports/<name>/<frame>` so a reader can go from a count in the document to the rows
    behind it.

    Common Usage:
        ```python
        from workbench.reports import Report, sections
        from workbench.utils.pandas_utils import compare_keyed, data_quality

        report = Report("Mouse PK intake", summary="The collected arms match the DataSource.")
        report.add(sections.data_quality(data_quality(collected, key="batch")))
        report.add(sections.comparison(compare_keyed(collected, existing, on="batch"), "archive", "DataSource"))
        report.save("mouse_pk_intake.pdf")
        ```
    """

    def __init__(
        self,
        title: str,
        name: Optional[str] = None,
        summary: Optional[str] = None,
        filters: Optional[List[str]] = None,
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
        """
        self.filters = list(filters or [])
        self.notes: List[str] = []
        self._footnotes: Dict[int, List[str]] = {}
        self.title = title
        self.summary = summary
        self._cards: List[Summary] = []
        self.name = name or re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        self.frames: Dict[str, pd.DataFrame] = {}
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
        self.notes.extend(note for note in section.notes if note not in self.notes)
        self._pages.extend(section.pages)
        self.frames.update(section.frames)
        if section.summary is not None:
            self._cards.append(section.summary)
        return self

    def new_page(self, headline: str, sub: Optional[str] = None) -> Tuple[plt.Figure, float]:
        """Start a page: the report title as the kicker, then the headline and standfirst.

        Args:
            headline (str): The page's one idea.
            sub (Optional[str]): A sentence under the headline.

        Returns:
            Tuple[plt.Figure, float]: The page figure and the `y` to start the body at.
        """
        fig = pages.new_page()
        return fig, pages.head(fig, self.title, headline, sub)

    def location(self, frame: str) -> Optional[str]:
        """Where a detail frame is published, or None when the report is saved without publishing.

        Args:
            frame (str): The frame's name in its section.

        Returns:
            Optional[str]: The DataFrame Store location, for `DFStore().get(location)`.
        """
        return f"/reports/{self.name}/{frame}" if self._published else None

    def mark(self, note: str) -> str:
        """The marker for a note, e.g. "[1]", to put in the text it qualifies.

        Args:
            note (str): A note a section of this report declared.

        Returns:
            str: The note's number in brackets. Notes are numbered in the order they were added;
                one no section declared is added here.
        """
        if note not in self.notes:
            self.notes.append(note)
        return f"[{self.notes.index(note) + 1}]"

    def footnote(self, fig: plt.Figure, note: str) -> str:
        """Print a note at the foot of a page, and return its marker for the text above.

        Args:
            fig (plt.Figure): The page that refers to the note.
            note (str): A note a section of this report declared.

        Returns:
            str: The note's marker, as `mark` gives it.
        """
        on_page = self._footnotes.setdefault(id(fig), [])
        if note not in on_page:
            on_page.append(note)
        return self.mark(note)

    def _draw_footnotes(self, fig: plt.Figure) -> None:
        """The notes a page refers to, stacked up from just above the footer in note order.

        A page body leaves room for FOOTNOTE_LINES lines. Notes that need more are cut to a
        line each, and past that to one line naming them; the summary page has each in full.
        """
        notes = sorted(self._footnotes.pop(id(fig), []), key=self.notes.index)
        lines = [line for note in notes for line in textwrap.wrap(f"{self.mark(note)}  {note}", FOOTNOTE_WIDTH)]
        if (
            len(lines) > FOOTNOTE_LINES
        ):  # more than the page keeps room for: point to the summary, where each is in full
            room = FOOTNOTE_WIDTH - len(IN_FULL)
            lines = [
                textwrap.shorten(f"{self.mark(note)}  {note}", room, placeholder="...") + IN_FULL for note in notes
            ]
        if len(lines) > FOOTNOTE_LINES:
            marks = ", ".join(self.mark(note) for note in notes)
            lines = [f"Notes {marks} apply here, each in full on page 1."]
        for row, line in enumerate(reversed(lines)):
            fig.text(pages.L, FOOTNOTE_BASE + row * FOOTNOTE_LEAD, line, fontsize=7.8, color=pages.BODY)

    def build(self, pdf) -> None:
        """Draw every page onto a PdfPages-like object, numbering them in order.

        Args:
            pdf: An object with `savefig(fig)`.
        """
        if not self.summary and not self._cards:
            raise ValueError("A report needs a summary: pass summary= or add a section that has one")
        published = {name: df for name, df in self.frames.items() if self._published and not df.empty}
        # The summary is drawn last, once every page has named its notes, and saved first
        with plt.rc_context({"figure.max_open_warning": 0}):
            body = []
            for page in self._pages + ([self._data_page] if published else []):
                drawn = page(self)
                body += drawn if isinstance(drawn, list) else [drawn]
            for number, fig in enumerate(self._summary_pages() + body, start=1):
                self._draw_footnotes(fig)
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

    def _summary_pages(self) -> List[plt.Figure]:
        """The summary: the author's sentence, the filters, the notes, then a card per section.

        Laid out from the top down by what is actually there, onto as many pages as it takes.
        """
        fig, y = self.new_page("Summary", self.summary)
        figs = [fig]

        def turn() -> None:
            nonlocal fig, y
            fig, y = self.new_page("Summary, continued")
            figs.append(fig)

        def block(label: str, entries: List[str]) -> None:
            nonlocal y
            lines = [wrapped for entry in entries for wrapped in textwrap.wrap(entry, SUMMARY_WIDTH) or [""]]
            start = 0
            while start < len(lines):
                room = int((y - SUMMARY_BOTTOM - 0.03) // SUMMARY_LEAD)
                if room < min(3, len(lines) - start):
                    turn()
                    continue
                y = pages.sec(fig, y, label if start == 0 else f"{label}, continued", pages.MUTED)
                text = "\n".join(lines[start : start + room])
                y = pages.para(fig, y, text, size=9.2, width=SUMMARY_WIDTH, lead=SUMMARY_LEAD, gap=0.0)
                start += room
            pages.rule(fig, y - 0.004, c=pages.HAIRLINE)
            y -= 0.028

        if self.filters:
            block("What this report covers", [f"-  {line}" for line in self.filters])
        if self.notes:
            block("Notes", [f"{self.mark(note)}  {note}" for note in self.notes])
        for card in self._cards:
            if y - CARD_HEIGHT < SUMMARY_BOTTOM:
                turn()
            top = y
            y = pages.sec(fig, y, card.label, pages.MUTED)
            y = pages.para(fig, y, card.headline, size=11.5, width=56, color=pages.INK, lead=0.019)
            if card.stats:
                pages.stat_row(fig, pages.L, y - 0.004, [(k, str(v)) for k, v in card.stats[:4]], dx=0.11)
            if card.chart is not None:
                ax = fig.add_axes([0.63, top - CARD_HEIGHT + 0.045, 0.30, CARD_HEIGHT - 0.06])
                card.chart(ax)
                _thumbnail(ax)
            y = top - CARD_HEIGHT
            pages.rule(fig, y + 0.012, c=pages.HAIRLINE)
            y -= 0.012
        return figs

    def _data_page(self, _report: "Report") -> plt.Figure:
        """The closing page: every published frame, its size, and where to get it."""
        fig, y = self.new_page(
            "Data behind this report", "Every count in this report has its rows in the DataFrame Store."
        )
        rows = [[name, f"{len(df):,}", self.location(name)] for name, df in self.frames.items() if not df.empty]
        y = pages.table(fig, y, ["frame", "rows", "location"], rows, aligns=["left", "right", "left"])
        y = pages.sec(fig, y - 0.02, "Reading one", pages.MUTED)
        pages.para(fig, y, f'from workbench.api import DFStore\ndf = DFStore().get("{rows[0][2]}")', size=8.8)
        return fig
