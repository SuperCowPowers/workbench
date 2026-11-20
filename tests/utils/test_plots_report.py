"""Tests for the report module's axes styling and its lint-then-write saver."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

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
