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
    assert page_count(report) == 4  # the summary, the overview, the scatter of what disagrees, the rows


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
    left = pd.concat([left, left.iloc[:8].assign(cmax=-1.0)]).reset_index(drop=True)
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
    assert "35 of 35 paired rows agree. Needs a look: 5 rows only in archive." == report._cards[2].headline


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


def axes_per_page(report: Report) -> list:
    class Axes:
        found = []

        def savefig(self, fig, **_):
            self.found.append(len(fig.axes))

    axes = Axes()
    axes.found = []
    report.build(axes)
    return axes.found


def test_the_agreement_scatter_is_drawn_only_when_values_disagree():
    left = predictions(40)
    right = left.copy()
    agreeing = Report("Same").add(sections.comparison(compare_keyed(left, right, on="id", values=["cmax"])))
    right.loc[right.index[:4], "cmax"] *= 3
    disputed = Report("Differs").add(sections.comparison(compare_keyed(left, right, on="id", values=["cmax"])))

    # the summary thumbnail and the row-fate bars, plus the scatter when a value disagrees
    assert sum(axes_per_page(agreeing)) == 2
    assert sum(axes_per_page(disputed)) == 3


def test_comparison_section_handles_flags_text_and_unsigned_values(tmp_path):
    left = pd.DataFrame({"id": list("abcd"), "flag": [True, False, True, True], "note": [1, 2, 3, 4]})
    left["n"] = np.array([1, 8, 5, 5], dtype="uint64")
    right = left.assign(flag=[True, True, True, False], note=["1", "2", "x", "4"])
    right["n"] = np.array([2, 4, 5, 5], dtype="uint64")
    section = sections.comparison(compare_keyed(left, right, on="id"), "old", "new")

    mismatched = section.frames["comparison_mismatched"]
    assert list(mismatched[mismatched["column"] == "n"]["id"]) == ["b", "a"]  # relative gaps 1.0 then 0.5
    assert set(mismatched["column"]) == {"flag", "note", "n"}
    assert Report("Mixed").add(section).save(tmp_path / "report.pdf", publish=False) == CLEAN


def test_duplicate_rows_alone_still_get_a_chart_and_the_headline():
    df = pd.DataFrame({"x": [1.0, 1.0, 2.0, 2.0, 3.0], "y": [5.0, 5.0, 6.0, 6.0, 7.0]})
    card = sections.data_quality(data_quality(df)).summary
    assert card.chart is not None and "(whole row) has the most flagged rows (4)" in card.headline


def test_comparison_section_sorts_rows_into_fine_and_needs_a_look():
    left = predictions(30).assign(route=["po"] * 26 + ["ip"] * 4)
    left = pd.concat([left, left.iloc[[0, 1]]], ignore_index=True)
    right = pd.concat([left.iloc[:26], left.iloc[[0]]], ignore_index=True)
    right.loc[right.index[-1], "cmax"] *= 3  # the second "c0" row now differs; "c1" repeats the same value
    result = compare_keyed(left, right, on="id", values=["cmax"], scope=["route"], repeats="values")
    section = sections.comparison(result, "old", "new")

    assert list(section.frames["comparison_repeated_differ"]["id"]) == ["c0"]
    assert {"cmax in old", "cmax in new"} <= set(section.frames["comparison_repeated_differ"].columns)
    assert section.frames["comparison_left_out_of_scope"]["scope_reason"].unique().tolist() == ["route = ip"]
    assert section.summary.headline == ("26 of 26 paired rows agree. Needs a look: 1 repeated key whose values differ.")


def test_identical_frames_with_repeated_keys_need_nothing():
    df = pd.concat([predictions(10), predictions(10)], ignore_index=True)
    section = sections.comparison(compare_keyed(df, df, on="id"))
    assert section.summary.headline == "20 of 20 paired rows agree. Nothing needs a look."
    assert len(section.pages) == 1


def page_text(report: Report) -> list:
    """All the text on each page, in order."""

    class Text:
        found = []

        def savefig(self, fig, **_):
            self.found.append(" | ".join(text.get_text() for text in fig.texts))

    text = Text()
    text.found = []
    report.build(text)
    return text.found


def test_filters_are_listed_on_the_summary_page_and_take_a_cards_room(tmp_path):
    def page(report):
        return report.new_page("Method")[0]

    report = Report("Filtered", filters=["Species: mouse", "Route: PO and IV only"])
    for n in range(4):
        report.add(Section(pages=[page], summary=Summary(f"Step {n}", f"Finding number {n}.")))

    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    text = page_text(report)
    assert "Species: mouse" in text[0] and "Route: PO and IV only" in text[0]
    assert "Finding number 2." in text[0] and "Finding number 3." not in text[0] and "Finding number 3." in text[1]


def test_the_comparison_page_states_how_rows_were_compared():
    left = predictions(30).assign(route=["po"] * 26 + ["ip"] * 4)
    result = compare_keyed(left, left.iloc[:26], on="id", values=["cmax"], rtol=0.02, scope=["route"], repeats="values")
    text = page_text(Report("Method").add(sections.comparison(result, "old", "new")))[1]

    assert "matched on | id" in text
    assert "cmax; numbers agree within 0.02 of the new value" in text
    assert "limited to route | po" in text
    assert "however many rows carry it" in text


def rows_to_look_at(report: Report) -> list:
    """The text of each rows-to-look-at page."""
    return [page for page in page_text(report) if "rows to look at" in page]


def test_rows_to_look_at_show_ten_per_bucket_and_say_how_many_there_are():
    left = predictions(120)
    right = left.iloc[30:].copy()
    right.loc[right.index[:8], "cmax"] *= 2
    report = Report("Rows").add(sections.comparison(compare_keyed(left, right, on="id", values=["cmax"]), "old", "new"))
    report._published = True
    text = rows_to_look_at(report)[0]

    assert all(f"c{n}" in text for n in range(30, 38))  # all eight mismatches are on the page
    assert "Showing the first 10 of 30. All 30 rows: DFStore /reports/rows/comparison_left_only" in text
    assert "All 8 rows: DFStore /reports/rows/comparison_mismatched" in text


def test_the_not_expected_table_names_its_frame_without_a_showing_line():
    left = predictions(30).assign(route=["po"] * 26 + ["ip"] * 4)
    report = Report("Rows").add(sections.comparison(compare_keyed(left, left.iloc[:26], on="id", scope=["route"])))
    report._published = True
    text = rows_to_look_at(report)[0]

    assert "route = ip | 4" in text
    assert "All 4 rows: DFStore /reports/rows/comparison_left_out_of_scope" in text and "Showing" not in text


def test_rows_to_look_at_run_onto_another_page_when_they_do_not_fit():
    left = predictions(200).assign(route=[f"r{i % 40}" for i in range(200)])
    right = left.iloc[100:].copy()
    right.loc[right.index[:50], "cmax"] *= 2
    right = pd.concat([right, right.iloc[:1].assign(id="new", route="elsewhere")])
    result = compare_keyed(left, right, on="id", values=["cmax"], scope=["route"])
    section = sections.comparison(result, "old", "new", max_rows=45)
    report = Report("Rows").add(section)

    rows_pages = rows_to_look_at(report)
    assert len(rows_pages) > 1 and all("Continued." in page for page in rows_pages[1:])
    assert lowest_text(report) > 0.045


def test_a_missing_value_is_shown_blank_not_as_nan():
    left = predictions(12)
    left.loc[0, "project"] = None
    left.loc[1, "cmax"] = np.nan
    text = page_text(Report("Blank").add(sections.comparison(compare_keyed(left, left.iloc[6:], on="id"))))[2]
    assert "c0" in text and "nan" not in text.lower() and "None" not in text


NOTE = "The DataSource carries corrections, so a mismatch against the archive can be the fix."


def test_notes_are_numbered_listed_on_the_summary_and_printed_where_marked(tmp_path):
    left = predictions(40)
    right = left.iloc[5:].copy()
    right.loc[right.index[:3], "cmax"] *= 2
    report = Report("Noted", filters=["Species: mouse"])
    result = compare_keyed(left, right, on="id", values=["cmax"])
    report.add(sections.comparison(result, "archive", "DataSource", notes={"mismatched": NOTE}))
    report.add(sections.data_quality(data_quality(messy(), key="id"), notes=[NOTE, "A second note."]))

    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    assert report.notes == [NOTE, "A second note."]  # one number per note, however many sections use it
    text = page_text(report)
    assert f"[1]  {NOTE}" in text[0] and "[2]  A second note." in text[0]
    overview, scatter, rows = text[1:4]
    assert "mismatched [1]" in overview and "MISMATCHED [1]" in rows
    assert all(f"[1]  {NOTE}" in page for page in (overview, scatter, rows))  # printed at the foot of each
    quality = next(page for page in text if "Data quality" in page and "NOTE [1]" in page)
    assert NOTE in quality and "NOTE [2]" in quality
    assert lowest_text(report) > 0.04


def test_a_note_for_an_unknown_bucket_is_refused():
    df = predictions(10)
    with pytest.raises(ValueError, match="No bucket named"):
        sections.comparison(compare_keyed(df, df, on="id"), notes={"mismatches": "typo"})


def test_footnotes_stay_inside_the_right_margin():
    from workbench.utils.plots import report as pages

    long_note = "A caveat long enough to need wrapping, " * 6
    left = predictions(20)
    right = left.copy()
    right.loc[right.index[:2], "cmax"] *= 2
    report = Report("Wide").add(
        sections.comparison(compare_keyed(left, right, on="id"), notes={"mismatched": long_note})
    )

    class Edges:
        worst = 0.0

        def savefig(self, fig, **_):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            notes = [text for text in fig.texts if text.get_fontsize() == 7.8]
            self.worst = max([self.worst] + [text.get_window_extent(renderer).x1 / fig.bbox.width for text in notes])

    edges = Edges()
    report.build(edges)
    assert 0 < edges.worst <= pages.R + 0.002


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--no-cov"])
