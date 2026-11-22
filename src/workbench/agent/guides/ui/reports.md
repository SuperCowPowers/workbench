# Reports

> multi-page PDF reports: structure that survives a human reader, and a modern look

A **report** is a different artifact from a plot. `plotting` covers one figure answering
one question on screen; this covers a document someone forwards to a colleague who was
not in the room. Build one only when the user asks for a report, a PDF, or something to
share — otherwise `plt.show()`.

## Ask where it goes

A report is a file on someone's disk, so the path is the user's call. Their report
directory (the Workspace block of your system prompt) is the default; with none set,
ask rather than using the current working directory. Before building, name the file
you intend to write, say where it will land, and check you are not about to replace
something:

```python
out_pdf = os.path.join(ConfigManager().get_config("BOSCO_REPORT_DIR"), "method1_bateman_report.pdf")
print(out_pdf, "| exists already?", os.path.exists(out_pdf))
```

If it already exists, stop and ask. A re-render that silently overwrites yesterday's
version is the one mistake a report author cannot undo, and "I regenerated it" is no
help once the old numbers are gone. When the write succeeds, report the **full path and
size** so the user can find it without hunting.

Save the script that builds the report next to it, under the same name. When the
report directory is in a git repo, both are tracked: the PDF is the record of what was
found, and the script is how it was made. Git keeps a full copy of every version of a
PDF, so keep them small — pass `rasterized=True` to a scatter with thousands of points.

## Structure: lead with the answer

The instinct is to narrate the work in the order it happened — method, then data, then
findings. That order bores the reader out of the document before the findings arrive.
Invert it:

1. **The answer, visually.** Page 1 is a summary: the conclusion in a sentence, then a
   card per topic with its finding, its key numbers and one chart. A reader should be able
   to stop after page 1 and be usefully informed.
2. **The caveats.** What is wrong with the data, how reliability was scored, and the one
   plot that proves it. Define every metric *here*, where it is first used.
3. **Method.** How the numbers were produced, which variants were tried, what was
   adopted and why.
4. **Detail.** One page per group/model/cohort, for the reader who got interested.

Rules that do most of the work:

- **Every report opens with a summary page.** `Report` builds it from the sections; you
  supply the one sentence only a person can write.
- **Every table of rows names its frame.** Under a table that lists rows, print how many
  are shown and the `DFStore` location of all of them ("Showing the first 10 of 778. All
  778 rows: DFStore /reports/<name>/<frame>"). The sections do this; `report.location(frame)`
  gives the location for a table of your own.
- **Lead with a picture.** Each section and each page opens with a chart or graphic; a
  table backs it up, it does not replace it. A page of numbers alone is a page skipped.
- **One idea per page.** If a page needs two headlines it is two pages.
- **No wall of prose.** Cap any text block at ~4 lines, give it a short uppercase label,
  and put a number or a figure next to it. A full page of paragraphs is skipped.

When showing exemplars, pick **one clean and one broken** rather than two clean ones. A
reader learns the failure mode from seeing it, not from a count in a table.

## Build it from sections

`workbench.reports` composes a report from sections, so the common cases are a few lines.
Each section turns an analysis result into pages, and carries the rows behind every count:

```python
from workbench.reports import Report, sections
from workbench.utils.pandas_utils import compare_keyed, data_quality

report = Report("Mouse PK intake", summary="The collected arms match the DataSource; 15 are new.")
report.add(sections.performance(pred_df, "cmax", by="project", key="batch", log=True, unit="ng/mL"))
report.add(sections.data_quality(data_quality(collected, key="batch", target="cmax")))
report.add(sections.comparison(compare_keyed(collected, existing, on="batch"), "Archive", "DataSource"))
report.save(out_pdf)
```

| section | takes | leads with | then |
|---|---|---|---|
| `sections.performance` | a predictions frame | predicted-vs-measured scatter, error histogram | breakout by a column, largest errors |
| `sections.data_quality` | `data_quality(df, key=...)` | flagged rows per column, by check | findings worst first, example rows per warning |
| `sections.comparison` | `compare_keyed(left, right, on=...)` | where every row went, per side; left-vs-right scatter | what is fine, what needs a look, and the rows that need it |

Page 1 is generated: `summary=` is your conclusion, and each section adds a card with its
finding, key numbers and lead chart. A report with neither is refused.

The PDF navigates: clicking a card opens its section, every later page has a `SUMMARY` link back,
and the sections are the PDF's bookmarks. A card can lead with one number and a status:
`Summary(..., hero=("1,319", "of 1,327 paired rows agree"), needs_action=True)`.

A comparison names where each side's data lives:
`sections.comparison(..., locations={"Archive": "s3://bucket/prefix"})` prints the location under
that side's name on the overlap chart and in full with how the rows were compared.

**Notes say what the sections cannot.** A caveat specific to this data goes in as a `Note`, with a
severity (`serious`, `warning`, `info`) and usually the artifact it is about:

```python
from workbench.reports import Note

note = Note("8 arms in the Upload sheets carry another arm's values.", severity="serious", about="Archive")
report = Report("Mouse PK intake", notes=[note])
report.add(sections.comparison(result, "Archive", "DataSource"))
report.add(sections.data_quality(quality, title="Archive data quality", about="Archive"))
```

The note is printed once, under a badge carrying its number in its severity's colour (red, dark
yellow, green), in the section about that artifact (the one
about it alone, else the first to involve it). Every other section that involves the artifact gets
a `Note [1]` badge that links to it, and the cards of those sections carry the badge; a
serious note makes them read "Needs a look". A note with no `about` is printed on the summary
page. A page of your own shows its notes with `y = report.note_block(fig, y, ["Archive"])`.

**Break one-sided rows down.** `sections.comparison(..., by="year")` draws, beside the overlap,
how the rows held by one side only spread over a column, which shows where a gap concentrates.

**State what the numbers cover.** Pass `filters=[...]`, one line per restriction (species,
route, date range, rows left out and why), and they are listed on page 1. A reader who
does not know a filter was applied reads every count wrong. The comparison section states
its own method the same way: the key, the values and tolerance, the scope values, and how
rows sharing a key were judged.

`log=True` on the performance section is for positive quantities such as PK parameters: log
axes, metrics on log10 values, and GMFE with the share within 2x, 3x and 10x.

`report.save` lints, writes the PDF, and publishes each section's detail frames to
`/reports/<name>/<frame>` in the `DFStore`, replacing the previous run's. A closing page
lists them, so a reader goes from "7 mismatched" to the seven rows. `publish=False` skips
the store. Run the analysis functions on their own when the user wants the numbers rather
than a document.

Model performance for an endpoint is one call:

```python
from workbench.reports.model_performance import model_performance_report

report = model_performance_report("abalone-regression", df, out_pdf, by="sex", key="auto_id")
report.frames["performance_breakout"]      # the metrics table behind the pages
```

A page the sections do not cover is a function you add alongside them. It starts with
`report.new_page(headline)`; the footer, page number, axes fit and save are done for you.
Wrap it in a `Section` with a `Summary` to give it a card on page 1:

```python
from workbench.reports import Section, Summary
from workbench.utils.plots import report as pages

def page_method(report):
    fig, y = report.new_page("How the archive was read")
    y = pages.para(fig, y, "Each workbook carries its parameters twice ...")
    return fig

card = Summary("Method", "413 workbooks were read; 12 only after peeling a revision suffix.", [("Workbooks", 413)])
report.add(Section(pages=[page_method], summary=card))
```

The section charts are also usable on their own, each taking an optional `ax`:
`workbench.utils.plots.performance` (`pred_vs_actual`, `residuals`, `breakout_bars`) and
`workbench.utils.plots.data` (`overlap_venn`, `one_sided_bars`, `quality_bars`, `outlier_box`).

The rest of this guide is the craft behind those pages, for the ones you write yourself.

## Modern PDFs

The default matplotlib look — DejaVu Sans, black text, full box spines, grey grid —
reads as derpy. Four changes fix almost all of it: a real font pairing, softened ink,
hairline rules, and whitespace.

### Start from the shared module

`workbench.utils.plots.report` carries the look and the layout helpers, so a new report
is page functions rather than 200 lines of styling. Reach for it before hand-rolling.

```python
from workbench.utils.plots import report

report.theme()          # rcParams: fonts, STIX maths, fonttype 42, softened ink
```

It gives you the palette (`INK BODY MUTED RULE HAIRLINE ACCENTS`), the margins (`L R`),
page helpers that each take and return the running `y` cursor — `new_page`, `head`,
`sec`, `para`, `equation`, `table`, `stat_row`, `stat_tiles`, `panel`, `rule`, `footer`, `navigate` — plus `EQ` (typeset
PK and metric equations), `tidy` for axes, `text_width` for measuring, `fit_axes` / `audit`
for linting, and `save` to lint and write.

### Fonts and portability

**A report must render on a machine that is not yours.** Avenir Next and Helvetica Neue
are macOS-only; a hardcoded family looks right on one laptop and silently degrades to
DejaVu everywhere else. `report.theme()` resolves a preference chain and returns what it
actually got, so the fallback is a known quantity rather than a surprise:

```python
report.theme()
# {'heading': 'Avenir Next', 'body': 'Helvetica Neue', 'math': 'stix'}   on macOS
# {'heading': 'DejaVu Sans', 'body': 'DejaVu Sans',    'math': 'stix'}   on a bare Linux box
```

Note matplotlib often registers only one weight per macOS `.ttc`, so bold-vs-regular
comes from *switching family* rather than from `weight="bold"` — which is why the module
keeps `HEADING_FONT` and `BODY_FONT` separate.

### Equations

Fake equations with carets and slashes are the clearest tell of a thrown-together
document, and nested ones — `10 ^ median | log10(pred/obs) |` — are simply unreadable.
Use mathtext:

```python
y = report.equation(fig, y, report.EQ["bateman"], size=17)
y = report.equation(fig, y, r"$k_e = \dfrac{CL \cdot 0.06}{V_d}$")
```

`report.theme()` sets `mathtext.fontset="stix"`, and **STIX ships inside matplotlib** —
so equations render identically everywhere with no font install, which is exactly why it
beats the prettier option of mapping maths onto the body font. Serif maths against a sans
body is the normal journal pairing, not a mismatch.

`usetex=True` is a trap for a shared tool: it needs a full LaTeX plus `dvipng` on every
machine that renders the document. Check `shutil.which("latex")` before assuming it, and
prefer mathtext — it parses `\frac`, `\dfrac`, `\sqrt`, `\left(`, subscripts,
superscripts and greek, which covers everything a PK or ML report needs.

Keep equations *near the number they explain* — the GMFE formula belongs beside the GMFE
definition, not in an appendix.

### Palette

```python
report.INK, report.BODY, report.MUTED, report.RULE   # #111827  #3F4651  #9AA1AC  #E4E7EB
```

Pure black on white is harsh at print size. Three greys give hierarchy without bold
everywhere: `INK` for headings and numbers, `BODY` for prose, `MUTED` for labels and
units. Rules at `RULE` or lighter — a visible table border is a 1990s table border.

Give each group or cohort an accent from `report.ACCENTS` and keep it on every page, so a
reader tracks a group by colour alone:

```python
GROUP_COLOR = dict(zip("ABCD", report.ACCENTS))
```

### Lay pages out in figure coordinates

Full-page composition is not a subplot grid. Use a blank portrait page, place everything
in figure coordinates, and let each helper advance the `y` cursor down the page:

```python
fig = report.new_page()
y = report.head(fig, "section two", "Data Issues", "Fit quality is driven by ...")
y = report.sec(fig, y, "How fit quality is measured", report.MUTED)
y = report.para(fig, y, "The geometric mean fold error over ...")
y = report.equation(fig, y, report.EQ["gmfe"], size=12.5)
y = report.table(fig, y, ["variant", "gmfe", "auc"], rows, hl_row=2)
report.footer(fig, "Workbench - Method 1 validation", 2)
pdf.savefig(fig); plt.close(fig)
```

Because every helper **returns the new `y`**, pages compose by chaining and a wording
change never breaks the layout below it.

`table` takes no column positions, by design. It measures every header and cell, gives each
column its natural width, and spreads the slack evenly across the gaps — label column flush
at `x0`, last column flush at `x1` — then infers alignment, ranging numeric columns right
and text left (`aligns` overrides when an id looks like a float). A hand-placed column is
the surest way to bury a number under its neighbour, which is why there is no way to place
one by hand. A table too wide for the page is fitted rather than left to run off it: the
body is set smaller, then the longest cells are trimmed with an ellipsis, and it warns only
if that still is not enough. Hairline
rules between rows and no verticals: ASCII tables and boxed grids are what make a PDF look
like a terminal dump.

### Figures inside a report

Strip the chartjunk that is tolerable on screen and distracting in print:

```python
report.tidy(ax, "What the reader should conclude")   # no top/right spines, hairline grid, left title
```

Left-align titles with the text column and make them the **conclusion**, not the
contents: "Fit quality degrades monotonically with flag load" beats "GMFE vs flags".
Hollow markers (`mfc="white"`, `mew=1.5`) keep observed points legible on top of a fitted
line, and a 7-10% alpha fill under a curve adds weight without obscuring anything.
Annotate values directly on the points and drop the legend when two series are obvious.

Model performance charts are ready-made in `workbench.utils.plots.performance`, and each
draws onto an axes you place:

```python
from workbench.utils.plots import performance

ax = fig.add_axes([0.10, y - 0.36, 0.38, 0.32])
performance.pred_vs_actual(df[target], df["prediction"], groups=df["project"], log=True, unit="ng/mL", ax=ax)
performance.residuals(df[target], df["prediction"], log=True, ax=ax2)
performance.breakout_bars(gmfe_by_project, reference=overall_gmfe, xlabel="GMFE", ax=ax3)
```

`log=True` is for positive quantities: log axes with the 2x band and 3x / 10x guides.

### Assembly: build takes a pdf-like, `report.save` writes it

```python
def build_report(pdf):
    for page in (page_groups, page_issues, page_method):
        page(pdf)                      # each ends in pdf.savefig(fig); plt.close(fig)

lint = report.save(out_pdf, build_report)   # audits, then writes; logs the path and size
```

`save` raises `FileExistsError` rather than replace a file, unless `overwrite=True`. The
split is what makes a report lintable: `build` can be handed a stand-in that measures
pages instead of writing them, so nothing is overwritten while you iterate.

### Lint every page -- none of this raises on its own

```python
report.fit_axes(fig, pad=9.0)      # per page, just before savefig
report.audit(build_report)         # {'overlaps': [], 'glyphs': []} means clean; save returns the same
```

Three silent failures. **Missing glyphs** -- nice faces carry small character sets and
matplotlib substitutes a visible box without complaint; `->` and `<-` are absent from both
Avenir Next and Helvetica Neue, while `<= x · - ~ ^2` are fine, and the text helpers swap
the known offenders via `ASCII_FALLBACKS`. **Collisions** -- there is no layout engine for
free figure text, since `tight_layout` and `constrained_layout` only manage axes.
**Axes decorations** -- a title sits above the rect you gave `add_axes` and tick labels
below it, so a rect that looks generous on screen collides in print; `fit_axes` shrinks each
rect until its `get_tightbbox` clears the surrounding text by `pad`.

Two details keep `audit` honest rather than noisy: it compares **overlap fraction of the
smaller box** (12%) rather than raw intersection, so a paragraph whose lines
abut is not flagged, and it drops tick labels that are not really drawn -- matplotlib keeps artists
for ticks outside the view at stale positions, which on one report was 18 of 23 hits.

### Check page bounds too -- `audit` does not

`audit` finds text hitting text, not content running off the page, and both edges fail
silently. Check every page after `audit`:

- **Bottom:** the lowest non-footer text clears the footer's top by a few points. A
  paragraph that wraps longer than planned slides into it.
- **Left / right:** every axes' `ax.get_tightbbox(renderer)` -- which includes tick
  labels, axis labels and `ax.text` value labels -- sits inside `report.L`..`report.R`.
  An `add_axes` rect places the *plot area*; long horizontal-bar category labels hang
  off its left by their full width. A rect that pins x0 to the margin pushes them off the
  page, and a rect padded for them by guesswork leaves a dead gap and a title indented
  away from the text column. Measure the overhang and solve for the rect instead: draw,
  read `tightbbox.x0 - position.x0`, set x0 = `L` minus that, and repeat until it settles.
- **Titles:** a `loc="left"` title starts at the plot area, not the labels. After pinning,
  move `ax._left_title` so it is flush with the text column.

Draw the canvas first, then `t.get_window_extent(renderer)` for `fig.texts` and
`ax.get_tightbbox(renderer)` for axes give the boxes in display units. A negative
clearance, or a box outside the margins, is an overrun.

### Measure, do not eyeball

Overlap is usually a guessed column position. `report.text_width(s, size, font)` gives the
rendered width in figure-fraction units, and `stat_row` uses it to treat `dx` as a *minimum*
pitch that widens to fit its own labels. A hardcoded pitch looks fine with short labels and
collides the moment the data changes: "MEDIAN GMFE" renders 48.6pt wide, overprinting a
36.7pt column on four cards at once.

### `loc="left"` titles are not `ax.title`

`set_title(loc="left")` does **not** write to `ax.title`. matplotlib keeps separate
`_left_title` and `_right_title` artists and leaves `ax.title` empty, so anything checking
only `ax.title` silently skips every left-aligned title -- which is the house style here.
`audit` checks all three. Worth knowing beyond reports: it defeats hand-rolled collision
checks and any code that reads a title back off an axes.

## Define the metrics in the document

A report leaves the session it was made in. Any acronym it uses — GMFE, AUC ratio, R², a
flag name — needs a one-line definition *in the PDF*, with the formula and a sentence on
how to read a value. "GMFE 1.50 means the typical point is off by 50% in either
direction" is worth more than the formula alone.

## Keep the builders in the session

Define one function per page and leave them in the namespace along with the copy in
dicts (`GROUP_DEF`, `FLAG_META`, ...). Re-rendering after a wording change is then the
last cell again, and the user can tune text without touching layout. Tell them which
names you left behind.

## Checklist

- Output path agreed with the user; nothing silently overwritten.
- Built from `workbench.reports` sections where one fits; no count without the rows behind it.
- Page 1 is the summary and answers the question; method is late, not first.
- Every filter applied to the data is listed in `filters=`.
- A caveat that changes how a number reads is a note, marked where it applies.
- Every section opens with a chart; no page is tables alone.
- Every table of rows says how many it shows and where the full frame is.
- Every page has one headline, and no text block exceeds ~4 lines.
- Built on `report.theme()`; no hardcoded font family.
- Equations typeset with `report.equation` / `EQ`, never faked with carets.
- Figure titles state conclusions and are left-aligned to the text column.
- `report.save` (or `report.audit(build_report)`) returns no overlaps and no glyph warnings.
- Lowest text on every page clears the footer; every axes' tight bbox sits inside the L/R margins.
- Every metric and flag defined inside the document.
- Exemplars include a failure, not only successes.
