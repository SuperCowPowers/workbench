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


@dataclass
class Section:
    """Pages for a report, and the detail frames behind them.

    Attributes:
        pages (List[Callable]): Each is called with the Report and returns one page figure.
            Start a page with `report.new_page(headline)`; the Report adds the footer and
            page number, fits the axes and saves it.
        frames (Dict[str, pd.DataFrame]): Detail frames by name, published to the DataFrame
            Store when the report is saved. A page gets a frame's location from
            `report.location(name)`.
    """

    pages: List[Callable] = field(default_factory=list)
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)


class Report:
    """Report: a PDF document built from sections, with the rows behind it in the DataFrame Store.

    Sections come from `workbench.reports.sections` (model performance, data quality, data
    comparison) or are your own page functions. Saving lints every page, writes the PDF, and
    publishes each section's detail frames to `/reports/<name>/<frame>` so a reader can go
    from a count in the document to the rows behind it.

    Common Usage:
        ```python
        from workbench.reports import Report, sections
        from workbench.utils.pandas_utils import compare_keyed, data_quality

        report = Report("Mouse PK intake")
        report.add(sections.data_quality(data_quality(collected, key="batch")))
        report.add(sections.comparison(compare_keyed(collected, existing, on="batch"), "archive", "DataSource"))
        report.save("mouse_pk_intake.pdf")
        ```
    """

    def __init__(self, title: str, name: Optional[str] = None):
        """Report Init

        Args:
            title (str): Shown above every page headline and in the footer.
            name (Optional[str]): Names the report's DataFrame Store subtree, /reports/<name>.
                Defaults to the title in lowercase with underscores.
        """
        self.title = title
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
        published = {name: df for name, df in self.frames.items() if self._published and not df.empty}
        for number, page in enumerate(self._pages + ([self._data_page] if published else []), start=1):
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
