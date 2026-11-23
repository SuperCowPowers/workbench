# Reports

> multi-page PDF reports composed from sections, with the rows behind every count in the DFStore

A **report** is a document someone forwards to a colleague who was not in the room; `plotting`
covers one figure answering one question on screen. Build one when the user asks for a report,
a PDF, or something to share.

## Where it goes

A report is a file on someone's disk, so the path is the user's call. Their report directory
(the Workspace block of your system prompt) is the default; with none set, ask. Name the file
before building and check it does not exist:

```python
out_pdf = os.path.join(ConfigManager().get_config("BOSCO_REPORT_DIR"), "mouse_pk_intake.pdf")
print(out_pdf, "| exists already?", os.path.exists(out_pdf))
```

`report.save` refuses to replace a file without `overwrite=True`. On success, give the user the
full path and size. Save the script that built the report next to it, under the same name.

## Report and sections

`workbench.reports` composes a report from sections. Each turns an analysis result into pages
and carries the rows behind every count:

```python
from workbench.reports import Report, sections
from workbench.utils.pandas_utils import compare_keyed, data_quality

report = Report("Mouse PK intake", summary="The collected arms match the DataSource; 15 are new.")
report.add(sections.comparison(compare_keyed(collected, existing, on="batch"), "Archive", "DataSource"))
report.add(sections.data_quality(data_quality(collected, key="batch"), title="Archive data quality", about="Archive"))
report.add(sections.performance(pred_df, "cmax", by="project", key="batch", log=True, unit="ng/mL"))
report.save(out_pdf)
```

| section | takes | shows |
|---|---|---|
| `sections.comparison` | `compare_keyed(left, right, on=...)` | how rows were compared, a Venn of the overlap, what is fine and what needs a look, then the rows to look at |
| `sections.data_quality` | `data_quality(df, key=...)` | findings worst first beside a box plot per column with outliers, then the rows behind each warning |
| `sections.performance` | a predictions frame | metric tiles, predicted vs measured, error histogram, breakout by a column, largest errors |
| `sections.archive` | `Archive.read(...)` | where the Archive is, files read and unreadable, the reasons |

- `Report(title, name=None, summary=None, filters=None, notes=None)`. `summary` is the one
  sentence only a person can write; `filters` lists what the numbers are limited to (species,
  route, rows left out and why), shown on page 1; `notes` see below.
- Page 1 is always a summary: the sentence, the filters, general notes, then one card per
  section. A report with no summary and no cards is refused.
- The PDF navigates: a card opens its section, every later page has a `SUMMARY` link back, and
  the sections are the PDF's bookmarks.
- `report.save(path, overwrite=False, publish=True)` lints every page, writes the PDF, and
  publishes each section's frames to `/reports/<name>/<frame>` in the `DFStore`, replacing the
  previous run's. Tables of rows say how many are shown and where all of them are; a closing
  page lists the frames. `publish=False` skips the store.
- Every section starts on a fresh page; a page that overflows continues with the same headline
  marked "Continued".

Section options worth knowing:

- `comparison(..., locations={"Archive": "s3://bucket/prefix"})` prints where each side lives,
  under its name on the Venn and in full in the method table. `by="year"` draws the one-sided
  rows broken down by a column beside the Venn. `max_rows` caps the rows shown per bucket.
- `compare_keyed(..., scope=["route"], rtol=0.02, repeats="values")`: rows whose scope values
  the other side never holds are set aside as "not expected"; `repeats="values"` judges a key
  held by several rows on its values, for raw data where row counts are loose.
- `performance(..., log=True)` is for positive quantities such as PK parameters: log axes,
  metrics on log10 values, GMFE and the share within 2x, 3x and 10x.
- `about=` on `data_quality` and `performance` names the artifact the frame came from, so the
  report's notes on it show there. A comparison is about both its sides.

An endpoint's performance is one call: `model_performance_report(endpoint_name, df, out_pdf, by=..., key=...)`
from `workbench.reports.model_performance`.

## Notes

A note says what the sections cannot: a caveat specific to this data.

```python
from workbench.reports import Note

note = Note("10 arms in the Upload sheets carry another arm's values.", severity="serious",
            about="Archive", details=corrected_arms)
report = Report("Mouse PK intake", notes=[note])
```

- `severity` is `serious`, `warning` or `info`; the note's `NOTE [n]` badge is red, dark yellow
  or green.
- A note `about` an artifact is printed once, in the section about that artifact alone (else
  the first to involve it). Every other section involving it gets a badged line that links to
  it, and their cards carry the badge; a serious note makes them read "Needs a look". A note
  with no `about` is printed on the summary page. A note about something no section covers is
  refused.
- `details` is a frame of the rows the note is about: the first 10 are tabled under the note,
  and all of them are published as `note_<n>`.

## Cards

Each section's card is a `Summary(label, headline, stats, chart, hero, needs_action, size)`.
`hero` is the one number it leads with and what it counts; `needs_action` gives it a "Needs a
look" or "Fine" status. `size` is `"full"`, `"half_width"` (two side by side, no chart) or
`"half_height"` (full width, hero and stats only); `sections.archive(..., card_size=...)` picks
its own, half height by default.

## Archive: a client's pile of files

`workbench.utils.archive.Archive` owns where a client's files are, a local copy, and running a
reader over each. The reader is the client's code: one file in, that file's rows out, raising
on a file it cannot read.

```python
from workbench.utils.archive import Archive

archive = Archive("s3://bucket/pk_studies", read_upload, pattern="*.zip")
archive.files()                                        # name, key, size, modified
archive.fetch("~/work/datasets/pk_studies")            # copies an S3 Archive locally, skipping what is there
result = archive.read("~/work/datasets/pk_studies", workers=8)
result.rows, result.unreadable                         # rows with source_file and source_row; files and reasons
report.add(sections.archive(result, archive))
```

- The location is an S3 prefix (a whole bucket works) or a local or network directory, which
  is read in place. Pass `boto3_session=` when the Workbench role cannot read the bucket.
- `name` (default "Archive") is what the comparison labels it and what `Note(about=...)` names;
  pass `locations={archive.name: archive.location}` to the comparison.
- With `workers > 1` the reader must be a module-level function.

## A section of your own

A section is a function that lays its blocks down with a `Paginator` and returns its pages.
Wrap it in a `Section` to give it frames, a card and the artifacts it is about:

```python
from workbench.reports import Paginator, Panel, Section, Summary

def method(report):
    flow = Paginator(report, "How the archive was read", "413 workbooks.")
    flow.notes(["Archive"])
    flow.tiles([("Workbooks", "413"), ("Unreadable", "2")])
    flow.para("Each workbook carries its parameters twice ...")
    flow.charts(0.2, Panel(0.2, 0.6, lambda ax: ax.hist(sizes)))
    flow.label("Unreadable workbooks")
    flow.table(["workbook", "reason"], rows, frame="unreadable", count=len(unreadable))
    return flow.figs

card = Summary("Method", "413 workbooks were read.", [("Workbooks", 413)], hero=("413", "workbooks"))
report.add(Section(pages=[method], frames={"unreadable": unreadable}, summary=card, about=["Archive"]))
```

The paginator turns the page when the next block does not fit, splits a table by rows (never
leaving fewer than three on a page), and keeps a `label` with the block it heads. Blocks:
`tiles`, `para`, `lines`, `label`, `heading` (a sub-part with a rule), `table` (with `frame=` and
`count=` it prints the rows line), `where`, `charts(height, *Panel(x0, width, draw))` for a row of
charts at one height, `notes`, `gap`, and `block(height, draw)` for anything else, where
`draw(fig, y)` draws at the cursor and returns the `y` to continue at. Footer, page number, axes
fit and save are done by the report.

The section charts work on their own, each taking an optional `ax`:
`workbench.utils.plots.performance` (`pred_vs_actual`, `residuals`, `breakout_bars`) and
`workbench.utils.plots.data` (`overlap_venn`, `one_sided_bars`, `quality_list`, `outlier_box`).

## Writing a report that reads

- **Lead with the answer.** Page 1 should let a reader stop there usefully informed; caveats
  next, method late, detail last.
- **Lead with a picture.** A section opens with a chart; a table backs it up.
- **Every count has its rows.** Publish the frame behind it; the sections do.
- **State every filter** in `filters=`; a reader who does not know one was applied reads every
  count wrong.
- **Define every metric in the document**, with a sentence on how to read a value ("GMFE 1.5
  means the typical point is off by 50% either way").
- **Titles state conclusions**, left-aligned: "Fit quality degrades with flag load", not
  "GMFE vs flags".
- **Short prose.** About four lines per block, with a label and a number or figure beside it.
- Exemplars: one clean and one broken, not two clean.

## Page craft (`workbench.utils.plots.report`)

The look and the low-level helpers the sections are built on. `report.theme()` sets fonts,
STIX maths and the softened palette (`INK BODY MUTED RULE HAIRLINE ACCENTS`); the margins are
`L R`. Helpers take and return the `y` cursor: `new_page`, `head`, `sec`, `para`, `equation`,
`table`, `stat_row`, `stat_tiles`, `panel`, `badge`, `rule`, `footer`; `tidy(ax, title)` styles
an axes, `text_width` measures, `navigate` adds links and bookmarks to a written PDF, and
`audit` / `save` lint and write a hand-built report.

Facts that bite:

- **Fonts.** `theme()` resolves a preference chain (Avenir Next, Helvetica Neue, ... DejaVu
  Sans) and returns what it got, so a report renders on a machine without the Mac faces.
  Headings are drawn bold explicitly: newer matplotlib registers every weight of a `.ttc` and
  would otherwise pick the regular one.
- **Equations.** Use mathtext (`report.equation(fig, y, report.EQ["gmfe"])`), never carets and
  slashes. STIX ships with matplotlib, so maths renders everywhere; `usetex=True` needs LaTeX on
  every machine.
- **Missing glyphs.** Nice faces lack `->` and `<-`; matplotlib draws a box without complaint.
  The text helpers swap the known offenders, and `audit` reports the rest.
- **Tables** take no column positions: every cell is measured, numeric columns range right, and
  a table too wide for the page is set smaller, then its longest cells trimmed.
- **Measure, do not eyeball.** `text_width(s, size, font)` gives a string's width in figure
  units; a guessed column pitch collides the moment the data changes.
- **Lint.** `audit` finds text hitting text and missing glyphs, not content off the page; the
  paginator keeps blocks above the page floor. `fit_axes` shrinks an axes until its titles and
  tick labels clear the text around it.
- **`loc="left"` titles** are `ax._left_title`, not `ax.title`, which stays empty; code reading
  a title back must check both.
- Pass `rasterized=True` to a scatter with thousands of points; it keeps the PDF small.
