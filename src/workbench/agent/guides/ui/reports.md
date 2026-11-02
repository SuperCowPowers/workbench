# Reports

> multi-page PDF reports: structure that survives a human reader, and a modern look

A **report** is a different artifact from a plot. `plotting` covers one figure answering
one question on screen; this covers a document someone forwards to a colleague who was
not in the room. Build one only when the user asks for a report, a PDF, or something to
share — otherwise `plt.show()`.

## Ask where it goes

A report is a file on someone's disk, so the path is the user's call — not the current
working directory by default. Before building, name the file you intend to write, say
where it will land, and check you are not about to replace something:

```python
out_pdf = os.path.join(os.getcwd(), "method1_bateman_report.pdf")
print(out_pdf, "| exists already?", os.path.exists(out_pdf))
```

If it already exists, stop and ask. A re-render that silently overwrites yesterday's
version is the one mistake a report author cannot undo, and "I regenerated it" is no
help once the old numbers are gone. When the write succeeds, report the **full path and
size** so the user can find it without hunting.

## Structure: lead with the answer

The instinct is to narrate the work in the order it happened — method, then data, then
findings. That order bores the reader out of the document before the findings arrive.
Invert it:

1. **The answer, visually.** The taxonomy, the ranking, the groups — as cards or panels
   with one exemplar figure each. A reader should be able to stop after page 1 and be
   usefully informed.
2. **The caveats.** What is wrong with the data, how reliability was scored, and the one
   plot that proves it. Define every metric *here*, where it is first used.
3. **Method.** How the numbers were produced, which variants were tried, what was
   adopted and why.
4. **Detail.** One page per group/model/cohort, for the reader who got interested.

Two rules that do most of the work:

- **One idea per page.** If a page needs two headlines it is two pages.
- **No wall of prose.** Cap any text block at ~4 lines, give it a short uppercase label,
  and put a number or a figure next to it. A full page of paragraphs is skipped.

When showing exemplars, pick **one clean and one broken** rather than two clean ones. A
reader learns the failure mode from seeing it, not from a count in a table.

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
`sec`, `para`, `equation`, `table`, `stat_row`, `rule`, `footer` — plus `EQ` (typeset
PK and metric equations) and `audit_glyphs`.

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
y = report.para(fig, y, "The median absolute fold error over ...")
y = report.equation(fig, y, report.EQ["gmfe"], size=12.5)
y = report.table(fig, y, ["variant", "gmfe", "auc"], rows, xs=[L, L+0.40, L+0.49], hl_row=2)
report.footer(fig, "Workbench - Method 1 validation", 2)
pdf.savefig(fig); plt.close(fig)
```

Because every helper **returns the new `y`**, pages compose by chaining and a wording
change never breaks the layout below it. Tables are positioned text with hairline rules
between rows, not a monospace block — ASCII tables are the main thing that makes a PDF
look like a terminal dump.

### Figures inside a report

Strip the chartjunk that is tolerable on screen and distracting in print:

```python
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(alpha=0.25, lw=0.6, color=RULE)
ax.tick_params(length=2.5, labelsize=8.2)
ax.set_title("What the reader should conclude", loc="left", pad=12)
```

Left-align titles with the text column and make them the **conclusion**, not the
contents: "Fit quality degrades monotonically with flag load" beats "GMFE vs flags".
Hollow markers (`mfc="white"`, `mew=1.5`) keep observed points legible on top of a fitted
line, and a 7-10% alpha fill under a curve adds weight without obscuring anything.
Annotate values directly on the points and drop the legend when two series are obvious.

### Assembly and the glyph trap

```python
from matplotlib.backends.backend_pdf import PdfPages

def build_report(path):
    with PdfPages(path) as pdf:
        for build in (page_groups, page_issues, page_method):
            build(pdf)              # each ends in pdf.savefig(fig); plt.close(fig)
```

Nice fonts have small character sets, and matplotlib **silently** substitutes a missing
glyph into a visible box. Arrows are the usual casualty: `->` and `<-` are absent from
both Avenir Next and Helvetica Neue, while `<= x · - ~ ^2` characters are generally fine.
Audit the whole document rather than eyeballing it:

```python
report.audit_glyphs(build_report, "/tmp/preflight.pdf")    # [] means clean
```

The text helpers already swap the known offenders via `report.ASCII_FALLBACKS`; anything
the audit still reports needs an entry there or a different character. Maths inside
`$...$` is left alone, since STIX has the symbols.

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
- Page 1 answers the question; method is late, not first.
- Every page has one headline, and no text block exceeds ~4 lines.
- Built on `report.theme()`; no hardcoded font family.
- Equations typeset with `report.equation` / `EQ`, never faked with carets.
- Figure titles state conclusions and are left-aligned to the text column.
- Glyph warnings audited to zero.
- Every metric and flag defined inside the document.
- Exemplars include a failure, not only successes.
