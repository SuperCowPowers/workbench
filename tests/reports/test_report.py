"""Tests for Report and its sections: pages build clean, counts carry their rows, frames publish"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from workbench.api import DFStore  # noqa: E402
from workbench.reports import Report, Section, Summary, sections  # noqa: E402
from workbench.utils.pandas_utils import compare_keyed, data_quality  # noqa: E402

CLEAN = {"overlaps": [], "glyphs": []}


def predictions(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": [f"c{i}" for i in range(n)], "project": rng.choice(["alpha", "beta", "gamma"], n)})
    df["cmax"] = 10 ** rng.uniform(0.5, 3.5, n)
    df["prediction"] = df["cmax"] * 10 ** rng.normal(0, 0.2, n)
    return df


def messy() -> pd.DataFrame:
    df = predictions(60)
    df.loc[[3, 9], "cmax"] = np.nan
    df.loc[5, "prediction"] = np.inf
    return pd.concat([df, df.iloc[[20]]], ignore_index=True)


def page_count(report: Report) -> int:
    class Counter:
        pages = 0

        def savefig(self, fig, **_):
            self.pages += 1

    counter = Counter()
    report.build(counter)
    return counter.pages


def test_name_defaults_to_a_slug_of_the_title():
    assert Report("Mouse PK: intake (v2)").name == "mouse_pk_intake_v2"
    assert Report("Anything", name="chosen").name == "chosen"


def test_performance_section_builds_clean_and_ranks_the_worst_rows(tmp_path):
    df = predictions()
    section = sections.performance(df, "cmax", by="project", key="id", log=True, unit="ng/mL")
    report = Report("Performance").add(section)

    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    assert page_count(report) == 3  # the summary, then the section's two pages
    rows = section.frames["performance_rows"]
    assert len(rows) == len(df)
    assert rows["log10_fold"].abs().is_monotonic_decreasing
    assert list(section.frames["performance_breakout"]["group"])[0] == "all"


def test_performance_section_without_breakout_or_key(tmp_path):
    section = sections.performance(predictions(), "cmax")
    assert Report("Performance").add(section).save(tmp_path / "report.pdf", publish=False) == CLEAN
    assert list(section.frames["performance_rows"].columns) == ["index", "cmax", "prediction", "error"]


def test_data_quality_section_adds_examples_only_for_warnings(tmp_path):
    dirty = Report("Quality").add(sections.data_quality(data_quality(messy(), key="id")))
    clean = Report("Quality").add(sections.data_quality(data_quality(predictions(), key="id")))

    assert dirty.save(tmp_path / "dirty.pdf", publish=False) == CLEAN
    assert clean.save(tmp_path / "clean.pdf", publish=False) == CLEAN
    assert page_count(dirty) == 3 and page_count(clean) == 2


def test_comparison_section_publishes_one_row_per_disagreeing_value(tmp_path):
    left = predictions(50)
    right = left.iloc[10:].copy()
    right.loc[right.index[:3], "cmax"] *= 2
    right.loc[right.index[0], "project"] = "delta"
    result = compare_keyed(left, right, on="id", values=["cmax", "project"])
    section = sections.comparison(result, "archive", "DataSource")

    mismatched = section.frames["comparison_mismatched"]
    assert list(mismatched.columns) == ["id", "column", "archive", "DataSource"]
    assert sorted(mismatched["column"]) == ["cmax", "cmax", "cmax", "project"]
    assert len(section.frames["comparison_left_only"]) == 10
    report = Report("Comparison").add(section)
    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    assert page_count(report) == 3  # the summary, then the section's two pages


def test_identical_frames_compare_on_one_page(tmp_path):
    df = predictions(20)
    report = Report("Comparison").add(sections.comparison(compare_keyed(df, df, on="id")))
    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    assert page_count(report) == 2


def test_a_page_function_is_a_section(tmp_path):
    def page(report):
        fig, y = report.new_page("A custom page")
        return fig

    report = Report("Custom", summary="One custom page.").add(page)
    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    assert page_count(report) == 2


def test_repeated_frame_names_are_refused():
    section = sections.data_quality(data_quality(messy(), key="id"))
    report = Report("Quality").add(section)
    with pytest.raises(ValueError, match="quality_rows"):
        report.add(sections.data_quality(data_quality(messy(), key="id")))
    report.add(sections.data_quality(data_quality(messy(), key="id"), name="second"))


def test_existing_pdf_is_kept_and_nothing_is_published(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"yesterday")
    report = Report("Quality", name="testing_report_refused").add(Section(frames={"rows": predictions(5)}))

    with pytest.raises(FileExistsError):
        report.save(path)
    assert path.read_bytes() == b"yesterday"
    assert not DFStore().check("/reports/testing_report_refused/rows")


def test_save_publishes_frames_lists_them_and_clears_emptied_ones(tmp_path):
    store = DFStore(path_prefix="/reports/testing_report_publish")
    try:
        first = Report("Publish", name="testing_report_publish")
        first.add(sections.data_quality(data_quality(messy(), key="id")))
        assert first.save(tmp_path / "first.pdf") == CLEAN

        assert first.location("quality_rows") == "/reports/testing_report_publish/quality_rows"
        assert len(store.get("/quality_rows")) == len(first.frames["quality_rows"])
        assert page_count(first) == 4  # the summary, two section pages and the data page

        # A clean regeneration removes the rows the previous run published
        second = Report("Publish", name="testing_report_publish")
        spotless = pd.DataFrame({"id": ["a", "b", "c", "d"], "x": [1.0, 2.0, 3.0, 4.0]})
        second.add(sections.data_quality(data_quality(spotless, key="id")))
        second.save(tmp_path / "second.pdf")
        assert not store.check("/quality_rows")
        assert page_count(second) == 2
    finally:
        DFStore().delete_recursive("/reports/testing_report_publish")


def lowest_text(report: Report) -> float:
    """The lowest body text on any page, in figure units (the footer sits at 0.028)."""

    class Lowest:
        y = 1.0

        def savefig(self, fig, **_):
            self.y = min([self.y] + [t.get_position()[1] for t in fig.texts if t.get_position()[1] > 0.03])

    lowest = Lowest()
    report.build(lowest)
    return lowest.y


def test_performance_section_tolerates_a_key_that_includes_the_breakout(tmp_path):
    section = sections.performance(predictions(), "cmax", by="project", key=["project", "id"])
    assert list(section.frames["performance_rows"].columns) == ["project", "id", "cmax", "prediction", "error"]
    assert Report("Performance").add(section).save(tmp_path / "report.pdf", publish=False) == CLEAN


def test_performance_rows_are_not_multiplied_by_repeated_index_labels():
    df = pd.concat([predictions(30), predictions(30)])
    rows = sections.performance(df, "cmax", key="id").frames["performance_rows"]
    assert len(rows) == 60 and rows["error"].abs().is_monotonic_decreasing


def test_performance_section_refuses_a_frame_with_nothing_to_score():
    with pytest.raises(ValueError, match="to score"):
        sections.performance(predictions(20).assign(cmax=np.nan), "cmax")


def test_a_full_breakout_page_stays_above_the_footer():
    df = predictions(400)
    df["project"] = [f"project_{i % 12}" for i in range(len(df))]
    report = Report("Performance").add(sections.performance(df, "cmax", by="project", key="id", log=True))
    report._published = True
    assert lowest_text(report) > 0.045


def test_six_full_comparison_buckets_stay_above_the_footer():
    left = predictions(120).assign(route=["po"] * 100 + ["ip"] * 20)
    right = predictions(120).assign(route=["po"] * 100 + ["sc"] * 20)
    left = pd.concat([left, left.iloc[:8]]).reset_index(drop=True)
    left.loc[left.index[20:40], "id"] = [f"only_left_{i}" for i in range(20)]
    right.loc[right.index[40:60], "id"] = [f"only_right_{i}" for i in range(20)]
    right.loc[right.index[60:70], "cmax"] *= 2
    result = compare_keyed(left, right, on="id", values=["cmax"], scope=["route"])
    assert (result.summary().drop(["pairs", "agree"]) > 6).all()

    report = Report("Comparison").add(sections.comparison(result, "archive", "DataSource"))
    report._published = True
    assert lowest_text(report) > 0.045


def test_a_failing_page_leaves_the_previous_pdf_and_published_frames(tmp_path, monkeypatch):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"yesterday")
    published = []
    monkeypatch.setattr(Report, "_publish_frames", lambda self: published.append(self.name))

    def broken(report):
        raise RuntimeError("page failed")

    report = Report("Broken", summary="Never written.").add(Section(frames={"rows": predictions(5)})).add(broken)
    with pytest.raises(RuntimeError):
        report.save(path, overwrite=True)
    assert path.read_bytes() == b"yesterday" and published == []


def page_headlines(report: Report) -> list:
    """The headline of every page, in order (the largest text on the page)."""

    class Headlines:
        found = []

        def savefig(self, fig, **_):
            self.found.append(max(fig.texts, key=lambda text: text.get_fontsize()).get_text())

    headlines = Headlines()
    headlines.found = []
    report.build(headlines)
    return headlines.found


def test_page_one_is_the_summary_with_a_card_per_section(tmp_path):
    report = Report("Intake", summary="The collected arms are usable.")
    report.add(sections.performance(predictions(), "cmax", by="project", key="id", log=True))
    report.add(sections.data_quality(data_quality(messy(), key="id")))
    left = predictions(40)
    report.add(sections.comparison(compare_keyed(left, left.iloc[5:], on="id"), "archive", "DataSource"))

    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    assert page_headlines(report)[0] == "Summary"
    assert [card.label for card in report._cards] == ["Model performance", "Data quality", "archive against DataSource"]
    assert "within 2x" in report._cards[0].headline and "weakest project" in report._cards[0].headline
    assert "35 of 35 paired rows agree" in report._cards[2].headline


def test_a_report_with_nothing_to_summarise_is_refused(tmp_path):
    def page(report):
        return report.new_page("A custom page")[0]

    with pytest.raises(ValueError, match="needs a summary"):
        Report("Custom").add(page).save(tmp_path / "report.pdf", publish=False)
    assert not (tmp_path / "report.pdf").exists()


def test_a_custom_section_brings_its_own_card_and_cards_run_onto_a_second_page(tmp_path):
    def page(report):
        return report.new_page("Method")[0]

    report = Report("Cards")
    for n in range(6):
        card = Summary(f"Step {n}", f"Finding number {n}.", [("Rows", n)], lambda ax: ax.plot([0, 1], [0, n]))
        report.add(Section(pages=[page], summary=card))

    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    assert page_headlines(report)[:3] == ["Summary", "Summary, continued", "Method"]


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--no-cov"])
