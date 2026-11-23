"""Paginator: lays a section's blocks down its pages, turning the page when the next block does not fit"""

import textwrap
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from workbench.utils.plots import report as pages

# The lowest a page body may reach, in figure units
FLOOR = 0.07

# A table in figure units: its header with the space under its last row, and each row
TABLE_HEAD, TABLE_ROW = 0.029, 0.0172

# A table is not started, or continued, with fewer rows than this at the foot of a page
MIN_TABLE_ROWS = 3

# The height of a label, of a heading, and of the line under a table saying where its rows are
LABEL, HEADING, WHERE = 0.021, 0.04, 0.03


class Panel(NamedTuple):
    """One chart in a row of charts.

    Attributes:
        x0 (float): Left edge of its axes, in figure units.
        width (float): Width of its axes.
        draw (Callable): Called with the axes to draw the chart.
        height (Optional[float]): Height of its axes. Defaults to the row's.
        lift (float): How far its axes sit above the row's baseline.
    """

    x0: float
    width: float
    draw: Callable
    height: Optional[float] = None
    lift: float = 0.0


class Paginator:
    """Paginator: lays a section's blocks down its pages, turning the page when the next block does not fit.

    A section makes one, hands it blocks from the top down, and returns its `figs`. A block that
    does not fit under the cursor starts a new page with the same headline, marked "Continued". A
    table is split across pages by rows, and a label stays with the block it heads.

    Common Usage:
        ```python
        def build(report):
            flow = Paginator(report, "How the archive was read", "413 workbooks.")
            flow.notes(["Archive"])
            flow.tiles([("Workbooks", "413"), ("Unreadable", "2")])
            flow.label("Unreadable workbooks")
            flow.table(["workbook", "reason"], rows, frame="unreadable", count=len(unreadable))
            return flow.figs

        report.add(Section(pages=[build], frames={"unreadable": unreadable}, about=["Archive"]))
        ```
    """

    def __init__(self, report, headline: str, sub: Optional[str] = None, floor: float = FLOOR):
        """Paginator Init

        Args:
            report (Report): The report the pages belong to.
            headline (str): The headline of every page of the section.
            sub (Optional[str]): A sentence under the headline of the first page.
            floor (float): The lowest a block may reach. Defaults to FLOOR.
        """
        self.report, self.headline, self.floor = report, headline, floor
        self.fig, self.y = report.new_page(headline, sub)
        self.figs: List[plt.Figure] = [self.fig]
        self._top = self.y  # where this page's body started
        self._label: Optional[Tuple[str, str, str]] = None  # a label waiting for the block it heads

    def turn(self) -> None:
        """Start the next page, with the same headline."""
        self.fig, self.y = self.report.new_page(self.headline, continued=True)
        self.figs.append(self.fig)
        self._top = self.y

    def _fits(self, height: float) -> bool:
        """Whether a block fits under the cursor. An empty page takes any block, so one too tall still lands."""
        waiting = LABEL if self._label else 0.0
        return self.y - waiting - height >= self.floor or self.y >= self._top

    def _draw_label(self, continued: bool = False) -> None:
        """Draw the waiting label at the cursor; a continued one keeps waiting for the next page."""
        if self._label is None:
            return
        text, color, short = self._label
        self.y = pages.sec(self.fig, self.y, f"{short}, continued" if continued else text, color)

    @staticmethod
    def _take(room: int, left: int) -> int:
        """How many of the `left` rows to put on this page, which has `room` for some.

        Rows are held back when the rest would otherwise be a stub of fewer than MIN_TABLE_ROWS
        on the next page.
        """
        room = max(room, 1)
        stub = left - room
        if 0 < stub < MIN_TABLE_ROWS and room - (MIN_TABLE_ROWS - stub) >= MIN_TABLE_ROWS:
            room -= MIN_TABLE_ROWS - stub
        return room

    def block(self, height: float, draw: Callable) -> None:
        """Place one block, on a new page if it does not fit here.

        Args:
            height (float): The room the block needs, used only to decide whether to turn the page.
            draw (Callable): Called as `draw(fig, y)` to draw it at the cursor. Returns the `y`
                to continue at, or None to continue `height` lower.
        """
        if not self._fits(height):
            self.turn()
        self._draw_label()
        self._label = None
        below = draw(self.fig, self.y)
        self.y = below if below is not None else self.y - height

    def gap(self, height: float) -> None:
        """Leave space before the next block; none at the top of a page."""
        if self.y < self._top:
            self.y -= height

    def label(self, text: str, color: str = pages.INK, short: Optional[str] = None) -> None:
        """A small uppercase label for the next block. It is drawn with that block, never left behind it.

        Args:
            text (str): The label.
            color (str): Its colour.
            short (Optional[str]): What to call it on a table's later pages, as "<short>, continued".
        """
        self._label = (text, color, short or text)

    def heading(self, text: str, keep: float = 0.12) -> None:
        """A heading for a part of the section, set off by a rule, with room kept for what follows.

        Args:
            text (str): The heading.
            keep (float): Room that must be left under it on the page, so it never sits alone.
        """

        self.gap(0.012)
        if not self._fits(HEADING + keep):
            self.turn()
        ruled = self.y < self._top  # at the top of a page the page's own rule is just above

        def draw(fig, y):
            if ruled:
                pages.rule(fig, y)
            fig.text(pages.L, y - 0.014, text, fontsize=12.5, color=pages.INK, va="top").set(
                fontname=pages.HEADING_FONT, fontweight=pages.HEADING_WEIGHT
            )
            return y - HEADING

        self.block(HEADING, draw)

    def para(self, text: str, size: float = 9.5, width: int = 112, lead: float = 0.0155, **style) -> None:
        """Wrapped prose.

        Args:
            text (str): The text; blank lines break paragraphs.
            size (float): Font size.
            width (int): Characters per line.
            lead (float): Pitch between lines.
            **style: Passed to `plots.report.para` (color, x, gap).
        """
        lines = sum(len(textwrap.wrap(block, width) or [""]) for block in text.split("\n"))
        self.block(
            lines * lead + 0.01, lambda fig, y: pages.para(fig, y, text, size=size, width=width, lead=lead, **style)
        )

    def lines(self, entries: Sequence[str], size: float = 9.2, width: int = 112, lead: float = 0.0155) -> None:
        """A list of entries, one or more lines each, continued onto further pages when it is long.

        Args:
            entries (Sequence[str]): The entries, each wrapped to `width`.
            size (float): Font size.
            width (int): Characters per line.
            lead (float): Pitch between lines.
        """
        wrapped = [line for entry in entries for line in textwrap.wrap(entry, width) or [""]]
        start = 0
        while start < len(wrapped):
            waiting = LABEL if self._label else 0.0
            room = int((self.y - waiting - self.floor) // lead)
            if room < min(MIN_TABLE_ROWS, len(wrapped) - start) and self.y < self._top:
                self.turn()
                continue
            room = self._take(room, len(wrapped) - start)
            self._draw_label(continued=start > 0)
            text = "\n".join(wrapped[start : start + room])
            self.y = pages.para(self.fig, self.y, text, size=size, width=width, lead=lead, gap=0.0)
            start += room
        self._label = None

    def tiles(self, pairs: Sequence[Tuple[str, str]]) -> None:
        """The section's headline numbers, as a row of tiles.

        Args:
            pairs (Sequence[Tuple[str, str]]): Label and value for each tile.
        """
        self.block(0.072, lambda fig, y: pages.stat_tiles(fig, y, pairs))

    def charts(self, height: float, *panels: Panel, above: float = 0.03, below: float = 0.045) -> None:
        """A row of charts at one height.

        Args:
            height (float): Height of the axes.
            *panels (Panel): The charts, left to right.
            above (float): Room over the axes, for a title.
            below (float): Room under the axes, for tick labels and a legend.
        """

        def draw(fig, y):
            base = y - above - height
            for panel in panels:
                panel.draw(fig.add_axes([panel.x0, base + panel.lift, panel.width, panel.height or height]))
            return base - below

        self.block(above + height + below, draw)

    def notes(self, about: Sequence[str]) -> None:
        """The report's notes on some artifacts: each in full where it belongs, else a line linking to it.

        Args:
            about (Sequence[str]): The artifacts this section is about.
        """
        for height, draw in self.report.note_blocks(list(about)):
            self.block(height, draw)

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence],
        frame: Optional[str] = None,
        count: Optional[int] = None,
        sample: bool = True,
        hl_row: Optional[int] = None,
        **style,
    ) -> None:
        """A table, split by rows across as many pages as it needs.

        Args:
            headers (Sequence[str]): Column headers.
            rows (Sequence[Sequence]): The rows to show.
            frame (Optional[str]): The detail frame the rows come from. Its DataFrame Store
                location is printed under the table when the report is published.
            count (Optional[int]): Rows in that frame.
            sample (bool): The rows are the frame's own, so say when not all are shown.
            hl_row (Optional[int]): A row to set in the heading face.
            **style: Passed to `plots.report.table` (aligns, x0, x1, size).
        """
        rows, start = list(rows), 0
        tail = WHERE if frame is not None else 0.0
        while start < len(rows) or (start == 0 and not rows):
            waiting = LABEL if self._label else 0.0
            room = int((self.y - waiting - TABLE_HEAD - tail - self.floor) // TABLE_ROW)
            if room < min(MIN_TABLE_ROWS, max(len(rows) - start, 1)) and self.y < self._top:
                self.turn()
                continue
            room = self._take(room, len(rows) - start)
            self._draw_label(continued=start > 0)
            hot = hl_row - start if hl_row is not None and start <= hl_row < start + room else None
            self.y = pages.table(self.fig, self.y, headers, rows[start : start + room], hl_row=hot, **style)
            start += room
            if not rows:
                break
        self._label = None
        if frame is not None:
            self.where(frame, count if count is not None else len(rows), shown=len(rows) if sample else None)

    def where(self, frame: str, count: int, shown: Optional[int] = None) -> None:
        """A muted line saying how many rows were shown and where every row of a frame is.

        Says nothing about the location when the report is unpublished, and nothing at all when
        every row is on the page and there is no location to give.

        Args:
            frame (str): The detail frame's name.
            count (int): Rows in the frame.
            shown (Optional[int]): Rows shown on the page, when they are the frame's own.
        """
        location = self.report.location(frame)
        parts = [f"Showing the first {shown:,} of {count:,}." if shown is not None and shown < count else ""]
        parts += [f"All {count:,} rows: DFStore {location}" if location is not None and count else ""]
        note = " ".join(part for part in parts if part)
        if note:
            self.y = pages.para(self.fig, self.y - 0.004, note, size=8.2, color=pages.MUTED) - 0.012
