"""Tests for Report and its sections: pages build clean, counts carry their rows, frames publish"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from workbench.api import DFStore  # noqa: E402
from workbench.reports import Note, Report, Section, Summary, sections  # noqa: E402
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
    assert "Rows to look at" in " ".join(page_text(dirty)) and "Rows to look at" not in " ".join(page_text(clean))


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
    assert page_count(report) == 3  # the summary, the overview, the rows


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
        assert page_count(first) == 3  # the summary, the section and the data page

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
    cards = [card for card, *_ in report._cards]
    assert [card.label for card in cards] == ["Model performance", "Data quality", "archive against DataSource"]
    assert "within 2x" in cards[0].headline and "weakest project" in cards[0].headline
    assert "35 of 35 paired rows agree. Needs a look: 5 rows only in archive." == cards[2].headline


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
    assert page_headlines(report)[:3] == ["Summary", "Summary", "Method"]
    assert "| CONTINUED |" in page_text(report)[1]


def axes_per_page(report: Report) -> list:
    class Axes:
        found = []

        def savefig(self, fig, **_):
            self.found.append(len(fig.axes))

    axes = Axes()
    axes.found = []
    report.build(axes)
    return axes.found


def test_a_breakdown_column_adds_a_chart_beside_the_overlap():
    left = predictions(40)
    result = compare_keyed(left, left.iloc[5:], on="id")
    plain = Report("Plain").add(sections.comparison(result))
    broken_down = Report("By project").add(sections.comparison(result, by="project"))

    # the summary thumbnail and the overlap, plus the breakdown when a column is named
    assert sum(axes_per_page(plain)) == 2
    assert sum(axes_per_page(broken_down)) == 3
    assert "Rows on one side only, by project" in axes_titles(broken_down)


def axes_titles(report: Report) -> list:
    class Titles:
        found = []

        def savefig(self, fig, **_):
            self.found += [ax.get_title(loc="left") for ax in fig.axes]

    titles = Titles()
    titles.found = []
    report.build(titles)
    return titles.found


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


def nothing_below_the_floor(report: Report, floor: float = 0.07) -> bool:
    """Whether every page keeps its text and its charts above the floor (the footer aside)."""

    class Lowest:
        ok = True

        def savefig(self, fig, **_):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            body = [t for t in fig.texts if t.get_text().strip() and t.get_position()[1] > 0.03]
            edges = [t.get_window_extent(renderer).y0 / fig.bbox.height for t in body]
            edges += [ax.get_tightbbox(renderer).y0 / fig.bbox.height for ax in fig.axes]
            self.ok = self.ok and min(edges, default=1.0) >= floor - 0.004

    lowest = Lowest()
    report.build(lowest)
    return lowest.ok


def test_a_crowded_data_quality_page_runs_onto_the_next():
    rng = np.random.default_rng(4)
    wide = pd.DataFrame({f"col{n}": 10 ** rng.normal(2, 1, 300) for n in range(12)})
    for n in range(12):
        wide.loc[rng.choice(300, 5 + n, replace=False), f"col{n}"] = np.nan
    wide.loc[0, ["col0", "col1", "col2"]] = 1e15
    wide = pd.concat([wide, wide.iloc[[1, 2]]], ignore_index=True)
    report = Report("Wide").add(sections.data_quality(data_quality(wide)))

    assert len(axes_titles(report)) == 1 + 3  # the summary card's chart and three box plots
    assert nothing_below_the_floor(report)


def test_a_note_with_details_does_not_push_a_full_page_past_the_floor():
    details = pd.DataFrame({"arm": [f"ID-{n}" for n in range(10)], "archive": [1.5] * 10, "corrected": [2.5] * 10})
    left = predictions(120).assign(year=["2021", "2022", "2023"] * 40)
    right = left.iloc[30:].copy()
    right.loc[right.index[:8], "cmax"] *= 2
    result = compare_keyed(left, right, on="id", values=["cmax"], rtol=0.02)
    report = Report("Noted", notes=[Note(NOTE, severity="serious", about="Archive", details=details)])
    report.add(sections.comparison(result, "Archive", "DataSource", by="year"))
    report.add(sections.performance(predictions(), "cmax", by="project", key="id", log=True, about="Archive"))

    assert nothing_below_the_floor(report)
    text = page_text(report)
    assert sum(page.count(NOTE) for page in text) == 1 and "on Archive: see page" in text[1]


def test_a_split_table_never_leaves_a_stub_of_rows_on_the_next_page():
    from workbench.reports import Paginator

    def long_table(rows: int) -> list:
        def build(report):
            flow = Paginator(report, "Long table")
            flow.table(["n", "square"], [[f"row{n}", n * n] for n in range(rows)])
            return flow.figs

        return page_text(Report("Stub", summary="One table.").add(build))

    full = next(rows for rows in range(20, 80) if len(long_table(rows + 1)) == 3)  # the most one page holds
    for over in (1, 2, 3):  # a row or two more than fits takes rows with it, so the next page has three
        text = long_table(full + over)
        assert len(text) == 3 and "| CONTINUED |" in text[2]
        assert f"row{full + over - 3} |" in text[2] and f"row{full + over - 4} |" not in text[2]


def test_every_built_in_section_stays_above_the_floor():
    report = Report("All three")
    report.add(sections.performance(predictions(), "cmax", by="project", key="id", log=True))
    report.add(sections.data_quality(data_quality(messy(), key="id")))
    left = predictions(40)
    report.add(sections.comparison(compare_keyed(left, left.iloc[5:], on="id"), "archive", "DataSource"))
    assert nothing_below_the_floor(report)


def rows_to_look_at(report: Report) -> str:
    """The text of a one-section report from its "Rows to look at" heading on."""
    text = " | ".join(page_text(report)[1:])
    return text[text.index("Rows to look at") :]


def test_rows_to_look_at_show_ten_per_bucket_and_say_how_many_there_are():
    left = predictions(120)
    right = left.iloc[30:].copy()
    right.loc[right.index[:8], "cmax"] *= 2
    report = Report("Rows").add(sections.comparison(compare_keyed(left, right, on="id", values=["cmax"]), "old", "new"))
    report._published = True
    text = rows_to_look_at(report)

    assert all(f"c{n}" in text for n in range(30, 38))  # all eight mismatches are on the page
    assert "Showing the first 10 of 30. All 30 rows: DFStore /reports/rows/comparison_left_only" in text
    assert "All 8 rows: DFStore /reports/rows/comparison_mismatched" in text


def test_the_not_expected_table_names_its_frame_without_a_showing_line():
    left = predictions(30).assign(route=["po"] * 26 + ["ip"] * 4)
    report = Report("Rows").add(sections.comparison(compare_keyed(left, left.iloc[:26], on="id", scope=["route"])))
    report._published = True
    text = rows_to_look_at(report)

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

    section_pages = page_text(report)[1:]
    assert len(section_pages) > 2 and all("| CONTINUED |" in page for page in section_pages[1:])
    assert ", CONTINUED" in " ".join(section_pages)  # a table split across pages says so
    assert nothing_below_the_floor(report)


def test_a_missing_value_is_shown_blank_not_as_nan():
    left = predictions(12)
    left.loc[0, "project"] = None
    left.loc[1, "cmax"] = np.nan
    text = rows_to_look_at(Report("Blank").add(sections.comparison(compare_keyed(left, left.iloc[6:], on="id"))))
    assert "c0" in text and "nan" not in text.lower() and "None" not in text


NOTE = "8 arms in the Upload sheets carry another arm's values."


def annotations(path) -> list:
    """For each page, the pages its links open (all counted from 1)."""
    reader = PdfReader(path)
    number = {page.indirect_reference: n for n, page in enumerate(reader.pages, start=1)}
    return [[number[a.get_object()["/Dest"][0]] for a in page.get("/Annots", [])] for page in reader.pages]


def noted_report(note: Note) -> Report:
    left = predictions(40)
    report = Report("Noted", notes=[note])
    report.add(sections.comparison(compare_keyed(left, left.iloc[5:], on="id"), "Archive", "DataSource"))
    report.add(sections.data_quality(data_quality(messy(), key="id"), title="Archive data quality", about="Archive"))
    return report


def test_a_note_is_printed_once_with_its_artifact_and_linked_from_the_other_sections(tmp_path):
    report = noted_report(Note(NOTE, severity="serious", about="Archive"))
    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN

    text = page_text(report)
    home = next(n for n, page in enumerate(text, start=1) if NOTE in page)
    assert "Archive data quality" in text[home - 1] and "NOTE [1] | on Archive" in text[home - 1]
    assert sum(page.count(NOTE) for page in text) == 1
    assert f"NOTE [1] | on Archive: see page {home}" in text[1]  # the comparison names the page that has it
    assert home in annotations(tmp_path / "report.pdf")[1]  # and its marker opens that page


def summary_badges(report: Report) -> list:
    """The fill colour of each badge on the summary page."""
    from matplotlib.colors import to_hex
    from matplotlib.patches import FancyBboxPatch

    class Badges:
        found = None

        def savefig(self, fig, **_):
            if self.found is None:  # a badge is the only filled patch that is not a panel
                fills = [to_hex(a.get_facecolor()) for a in fig.artists if isinstance(a, FancyBboxPatch)]
                self.found = [fill for fill in fills if fill != "#f6f7f9"]

    badges = Badges()
    report.build(badges)
    return badges.found


def test_a_serious_note_badges_its_cards_and_makes_them_need_a_look(tmp_path):
    serious = noted_report(Note(NOTE, severity="serious", about="Archive"))
    summary = page_text(serious)[0]
    assert summary.count("NOTE [1]") == 2 and summary.count("| Needs a look |") == 2 and "| Fine |" not in summary
    assert summary_badges(serious) == ["#b42318", "#b42318"]

    assert summary_badges(noted_report(Note(NOTE, severity="warning", about="Archive"))) == ["#a16207", "#a16207"]
    assert summary_badges(noted_report(Note(NOTE, severity="info", about="Archive"))) == ["#15803d", "#15803d"]


def test_a_note_lands_with_the_first_section_to_involve_its_artifact_when_none_is_about_it_alone():
    left = predictions(40)
    report = Report("Comparison only", notes=[Note(NOTE, about="Archive")])
    report.add(sections.comparison(compare_keyed(left, left.iloc[5:], on="id"), "Archive", "DataSource"))
    text = page_text(report)
    assert NOTE in text[1] and "see page" not in " ".join(text)


def test_a_general_note_is_printed_on_the_summary_page(tmp_path):
    def page(report):
        return report.new_page("Method")[0]

    report = Report("General", summary="One page.", notes=[Note("Read from the 2024 archive only.")]).add(page)
    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    text = page_text(report)
    assert "Read from the 2024 archive only." in text[0] and "NOTE [1]" in text[0]


def test_a_note_about_something_no_section_covers_is_refused():
    report = noted_report(Note(NOTE, about="FeatureSet"))
    with pytest.raises(ValueError, match="FeatureSet"):
        page_text(report)


def test_outliers_get_a_box_plot_on_the_quality_page():
    df = predictions(60)
    df.loc[0, "cmax"] = df["cmax"].max() * 1e6
    report = Report("Boxed").add(sections.data_quality(data_quality(df, key="id")))
    assert "cmax" in axes_titles(report)


def test_a_note_tables_its_details_and_publishes_them(tmp_path):
    swapped = pd.DataFrame({"arm": [f"ID-{n}" for n in range(14)], "archive": [1.5] * 14, "corrected": [2.5] * 14})
    report = noted_report(Note(NOTE, severity="serious", about="Archive", details=swapped))
    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN

    home = next(page for page in page_text(report) if NOTE in page)
    assert "ARM | ARCHIVE | CORRECTED" in home and "ID-9 | 1.5 | 2.5" in home and "ID-10" not in home
    assert "Showing the first 10 of 14." in home
    assert report.frames["note_1"] is swapped


def test_a_card_with_no_pages_to_open_is_refused():
    with pytest.raises(ValueError, match="no pages"):
        Report("Example").add(Section(summary=Summary("Result", "Done.")))


def test_a_note_detail_cell_holding_a_list_is_shown_as_text():
    details = pd.DataFrame({"arm": ["ID-1", "ID-2"], "doses": [[1, 2], [3]]})
    report = noted_report(Note(NOTE, about="Archive", details=details))
    home = next(page for page in page_text(report) if NOTE in page)
    assert "[1, 2]" in home


def test_a_long_note_fits_its_panel():
    from matplotlib.patches import FancyBboxPatch

    long_note = "A caveat that goes on at length about why this number misleads. " * 10
    report = noted_report(Note(long_note, about="Archive"))

    class Inside:
        ok = None

        def savefig(self, fig, **_):
            words = [t for t in fig.texts if t.get_text() and t.get_text() in long_note]
            if not words:
                return
            panels = [a for a in fig.artists if isinstance(a, FancyBboxPatch) and a.get_height() > 0.05]
            floor = min(panel.get_y() for panel in panels)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            lowest = min(t.get_window_extent(renderer).y0 / fig.bbox.height for t in words)
            self.ok = lowest >= floor

    inside = Inside()
    report.build(inside)
    assert inside.ok


def card_boxes(report: Report) -> list:
    """The (x0, y0, x1, y1) box of every card on the first summary page."""
    from matplotlib.patches import FancyBboxPatch

    class Boxes:
        found = None

        def savefig(self, fig, **_):
            if self.found is None:
                panels = [a for a in fig.artists if isinstance(a, FancyBboxPatch) and a.get_height() > 0.05]
                self.found = [
                    tuple(
                        round(v, 3)
                        for v in (a.get_x(), a.get_y(), a.get_x() + a.get_width(), a.get_y() + a.get_height())
                    )
                    for a in panels
                ]

    boxes = Boxes()
    report.build(boxes)
    return boxes.found


def test_mini_cards_pair_side_by_side_or_take_half_the_height(tmp_path):
    def page(report):
        return report.new_page("Method")[0]

    report = Report("Mini", summary="Three cards.")
    for size in ("half_width", "half_width", "half_height"):
        card = Summary(size, "A finding.", [("Rows", "3")], lambda ax: ax.plot([0, 1]), hero=("3", "rows"), size=size)
        report.add(Section(pages=[page], summary=card))
    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN

    left, right, short = card_boxes(report)
    assert left[1] == right[1] and left[2] < right[0]  # the two half-width cards share a row
    assert round(short[3] - short[1], 3) == round((left[3] - left[1]) / 2, 3)  # half the height
    assert short[0] == left[0] and short[2] == right[2]  # and the full width
    assert sum(axes_per_page(report)[:1]) == 0  # no card this small draws its chart
    assert annotations(tmp_path / "report.pdf")[0] == [2, 3, 4]  # each still opens its section


def test_an_unknown_card_size_is_refused():
    with pytest.raises(ValueError, match="size"):
        Summary("Result", "Done.", size="quarter")


def test_a_note_panel_has_no_room_for_a_rows_line_it_does_not_print():
    details = pd.DataFrame({"arm": ["ID-1", "ID-2"]})
    report = noted_report(Note(NOTE, about="Archive", details=details))
    unpublished = report._note_height(0)
    report._published = True
    assert report._note_height(0) - unpublished > 0.02


def test_an_unknown_severity_is_refused():
    with pytest.raises(ValueError, match="severity"):
        Note(NOTE, severity="critical")


def test_a_long_note_stays_inside_the_margins(tmp_path):
    report = noted_report(Note("A caveat that goes on at length about why this number misleads. " * 8, about="Archive"))
    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN

    class Edges:
        right = 0.0

        def savefig(self, fig, **_):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            edges = [t.get_window_extent(renderer).x1 / fig.bbox.width for t in fig.texts if t.get_text().strip()]
            self.right = max([self.right] + edges)

    edges = Edges()
    report.build(edges)
    assert edges.right <= 0.932 + 0.002


def body_text_floor(report: Report) -> float:
    """The lowest text on any page that is not the footer, in figure units."""

    class Lowest:
        y = 1.0

        def savefig(self, fig, **_):
            body = [text.get_position()[1] for text in fig.texts if text.get_fontsize() != 7.6]
            self.y = min([self.y] + body)

    lowest = Lowest()
    report.build(lowest)
    return lowest.y


def test_a_long_list_of_filters_pushes_cards_onto_the_next_summary_page(tmp_path):
    def page(report):
        return report.new_page("Method")[0]

    report = Report("Many filters", filters=[f"Filter number {n}" for n in range(20)])
    for n in range(3):
        report.add(Section(pages=[page], summary=Summary(f"Step {n}", f"Finding number {n}.")))

    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    text = page_text(report)
    assert "Filter number 19" in text[0] and "Finding number 2." not in text[0] and "Finding number 2." in text[1]
    assert body_text_floor(report) > 0.06


def test_more_filters_than_a_page_holds_continue_on_the_next(tmp_path):
    report = Report("Too many", summary="A lot is left out.", filters=[f"Filter number {n}" for n in range(80)])
    assert report.save(tmp_path / "report.pdf", publish=False) == CLEAN
    text = page_text(report)
    assert "Filter number 0" in text[0] and "Filter number 79" in text[1] and "COVERS, CONTINUED" in text[1]
    assert body_text_floor(report) > 0.06


def test_the_method_table_states_summed_tolerances():
    left = predictions(10)
    result = compare_keyed(left, left, on="id", values=["cmax"], rtol=0.02, atol=1)
    text = page_text(Report("Method").add(sections.comparison(result, "old", "new")))[1]
    assert "numbers agree within 1 plus 0.02 of the new value" in text


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--no-cov"])


def test_cards_open_their_sections_and_every_page_leads_back(tmp_path):
    report = Report("Intake", summary="The collected arms are usable.")
    report.add(sections.performance(predictions(), "cmax", by="project", key="id", log=True))
    report.add(sections.data_quality(data_quality(messy(), key="id")))
    report.save(tmp_path / "report.pdf", publish=False)

    reader = PdfReader(tmp_path / "report.pdf")
    headlines = page_headlines(report)
    quality = headlines.index("Data quality")

    def targets(page: int) -> list:
        dests = [a.get_object()["/Dest"][0] for a in reader.pages[page].get("/Annots", [])]
        return [i for dest in dests for i, p in enumerate(reader.pages) if p.indirect_reference == dest]

    assert targets(0) == [1, quality]
    assert all(targets(page) == [0] for page in range(1, len(reader.pages)))
    bookmarks = [(o.title, reader.get_destination_page_number(o)) for o in reader.outline]
    assert bookmarks == [("Summary", 0), ("Model performance", 1), ("Data quality", quality)]
    assert reader.metadata.title == "Intake"
    assert f"PAGE {quality + 1}" in page_text(report)[0]
