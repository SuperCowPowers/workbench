"""Tests for the report module's axes styling and its lint-then-write saver."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from workbench.utils.plots import report  # noqa: E402


def build(pdf, title: str = "A title", pages: int = 2):
    for n in range(1, pages + 1):
        fig = report.new_page()
        y = report.head(fig, "kicker", title)
        report.para(fig, y, "Body text.")
        report.footer(fig, "footer", n)
        pdf.savefig(fig)
        plt.close(fig)


def test_tidy_strips_spines_and_left_aligns_the_title():
    _, ax = plt.subplots()
    report.tidy(ax, "The conclusion")

    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert ax.get_title(loc="left") == "The conclusion" and ax.get_title() == ""
    plt.close("all")


def test_save_writes_the_pdf_and_returns_a_clean_lint(tmp_path):
    path = tmp_path / "nested" / "report.pdf"
    lint = report.save(path, build, pages=3)

    assert lint == {"overlaps": [], "glyphs": []}
    assert path.read_bytes().startswith(b"%PDF")


def test_save_refuses_to_replace_without_overwrite(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"yesterday")

    with pytest.raises(FileExistsError):
        report.save(path, build)
    assert path.read_bytes() == b"yesterday"

    report.save(path, build, overwrite=True)
    assert path.read_bytes().startswith(b"%PDF")


def test_save_reports_overlapping_text(tmp_path):
    def colliding(pdf):
        fig = report.new_page()
        fig.text(0.3, 0.5, "one line of text", fontsize=12)
        fig.text(0.3, 0.5, "another line here", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

    assert report.save(tmp_path / "report.pdf", colliding)["overlaps"]


def right_edges(fig) -> list:
    """The right edge of every piece of text on a page, in figure units."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [t.get_window_extent(renderer).x1 / fig.bbox.width for t in fig.texts if t.get_text().strip()]


def test_a_table_too_wide_for_the_page_is_fitted_not_overprinted(recwarn):
    headers = ["bat", "dose", "route", "matrix", "date", "archive rows", "datasource rows", "cmax in archive"]
    headers.append("cmax in datasource")
    row = ["ID-25163-14", "100", "po", "plasma", "2024-04-02", "3", "5", "133000; 166000; 185000"]
    row.append("104000; 128000; 133000; 166000; 185000; 201000; 244000")
    fig = report.new_page()
    report.table(fig, 0.8, headers, [row, row])

    assert max(right_edges(fig)) <= report.R + 0.002
    assert any(text.get_text().endswith("...") for text in fig.texts)
    assert "ID-25163-14" in [text.get_text() for text in fig.texts]  # short cells are left whole
    assert not [w for w in recwarn if "table:" in str(w.message)]
    plt.close(fig)


def test_a_table_that_fits_is_left_as_written():
    fig = report.new_page()
    report.table(fig, 0.8, ["check", "column", "rows"], [["missing", "dose", "3"], ["outliers", "cmax", "12"]])
    texts = [text.get_text() for text in fig.texts]

    assert not any(text.endswith("...") for text in texts)
    assert {text.get_fontsize() for text in fig.texts if text.get_text() in ("missing", "dose", "3")} == {8.8}
    plt.close(fig)


def test_a_table_is_still_fitted_when_its_widest_column_cannot_be_trimmed():
    """A short wide id cannot be cut, so the long text beside it gives up the room instead"""
    headers = ["id", "note", "more", "value"]
    row = ["WWWWWWWW", "i" * 160, "l" * 160, "12.5"]
    fig = report.new_page()
    report.table(fig, 0.8, headers, [row, row])
    texts = [text.get_text() for text in fig.texts]

    assert "WWWWWWWW" in texts and any(text.startswith("iii") and text.endswith("...") for text in texts)
    assert max(right_edges(fig)) <= report.R + 0.002
    plt.close(fig)


def test_stat_tiles_keep_every_value_inside_its_tile():
    fig = report.new_page()
    pairs = [("Rows", "1,400"), ("A much longer label than fits", "123,456,789,012,345"), ("R2", "0.85")]
    y = report.stat_tiles(fig, 0.8, pairs)

    assert y < 0.8 - 0.05
    assert max(right_edges(fig)) <= report.R
    plt.close(fig)


def test_navigate_adds_links_bookmarks_and_a_title(tmp_path):
    path = tmp_path / "three.pdf"
    with PdfPages(path) as pdf:
        for _ in range(3):
            fig = report.new_page()
            pdf.savefig(fig)
            plt.close(fig)
    report.navigate(path, [(0, (0.1, 0.5, 0.9, 0.7), 2)], [("First", 0), ("Last", 2)], "Three pages")

    reader = PdfReader(path)
    (link,) = [a.get_object() for a in reader.pages[0]["/Annots"]]
    assert [round(float(v)) for v in link["/Rect"]] == [61, 396, 551, 554]
    assert link["/Dest"][0] == reader.pages[2].indirect_reference
    assert [(o.title, reader.get_destination_page_number(o)) for o in reader.outline] == [("First", 0), ("Last", 2)]
    assert reader.metadata.title == "Three pages"
