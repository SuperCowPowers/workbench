"""Report: a PDF document built from sections, with the rows behind it in the DataFrame Store"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

from workbench.utils.plots import report as pages

log = logging.getLogger("workbench")

# Summary page layout: cards per page, and the height of one in figure units
CARDS_PER_PAGE, CARD_HEIGHT = 4, 0.185


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
        pages (List[Callable]): Each is called with the Report and returns one page figure.
            Start a page with `report.new_page(headline)`; the Report adds the footer and
            page number, fits the axes and saves it.
        frames (Dict[str, pd.DataFrame]): Detail frames by name, published to the DataFrame
            Store when the report is saved. A page gets a frame's location from
            `report.location(name)`.
        summary (Optional[Summary]): This section's card on the summary page.
    """

    pages: List[Callable] = field(default_factory=list)
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: Optional[Summary] = None


class Report:
    """Report: a PDF document built from sections, with the rows behind it in the DataFrame Store.

    Sections come from `workbench.reports.sections` (model performance, data quality, data
    comparison) or are your own page functions. Page 1 is always a summary: the author's
    sentence, then one card per section with its finding, key numbers and lead chart. Saving
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

    def __init__(self, title: str, name: Optional[str] = None, summary: Optional[str] = None):
        """Report Init

        Args:
            title (str): Shown above every page headline and in the footer.
            name (Optional[str]): Names the report's DataFrame Store subtree, /reports/<name>.
                Defaults to the title in lowercase with underscores.
            summary (Optional[str]): The author's conclusion, shown at the top of the summary page.
        """
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

    def build(self, pdf) -> None:
        """Draw every page onto a PdfPages-like object, numbering them in order.

        Args:
            pdf: An object with `savefig(fig)`.
        """
        if not self.summary and not self._cards:
            raise ValueError("A report needs a summary: pass summary= or add a section that has one")
        published = {name: df for name, df in self.frames.items() if self._published and not df.empty}
        summary_pages = [
            lambda _report, first=first: self._summary_page(first)
            for first in range(0, max(len(self._cards), 1), CARDS_PER_PAGE)
        ]
        every_page = summary_pages + self._pages + ([self._data_page] if published else [])
        for number, page in enumerate(every_page, start=1):
            fig = page(self)
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

    def _summary_page(self, first: int) -> plt.Figure:
        """A summary page: the author's sentence, then a card per section from `first` on."""
        fig, y = self.new_page("Summary" if first == 0 else "Summary, continued", self.summary if first == 0 else None)
        for card in self._cards[first : first + CARDS_PER_PAGE]:
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
        return fig

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
