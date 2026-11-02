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

### Fonts

Never assume a font exists; discover it, then pair a heading face with a body face.

```python
from matplotlib import font_manager
avail = {f.name for f in font_manager.fontManager.ttflist}
[c for c in ["Inter", "Helvetica Neue", "Avenir Next", "Roboto", "Lato", "DejaVu Sans"]
 if c in avail]
```

macOS reliably has **Avenir Next** and **Helvetica Neue**; Linux usually needs Inter or
Roboto installed and falls back to DejaVu Sans. Note that matplotlib often registers only
one weight per macOS `.ttc`, so bold-vs-regular comes from *switching family*
(`Avenir Next` for headings, `Helvetica Neue` for body) rather than from `weight="bold"`.

```python
plt.rcParams.update({
    "font.family": "Helvetica Neue",   # body
    "pdf.fonttype": 42,                # embed TrueType -> selectable, searchable text
    "axes.edgecolor": "#C9CED6", "axes.linewidth": 0.8,
    "text.color": "#111827", "axes.labelcolor": "#3F4651",
    "xtick.color": "#9AA1AC", "ytick.color": "#9AA1AC",
})
```

`pdf.fonttype: 42` matters — the default Type 3 output is not selectable or searchable.

### Ink, not black

```python
INK, BODY, MUTED, RULE = "#111827", "#3F4651", "#9AA1AC", "#E4E7EB"
```

Pure black on white is harsh at print size. Three greys give hierarchy without bold
everywhere: `INK` for headings and numbers, `BODY` for prose, `MUTED` for labels and
units. Rules at `#E4E7EB` or lighter — a visible table border is a 1990s table border.

Give each group/cohort an accent and use it consistently across every page, so a reader
tracks a group by color alone:

```python
GC = {"A": "#0F766E", "B": "#1D4ED8", "C": "#B45309", "D": "#BE123C"}
```

### Lay pages out in figure coordinates

Full-page composition is not a subplot grid. Use a blank portrait figure, place
everything in figure coordinates, and track a running `y` cursor down the page.

```python
L, R = 0.068, 0.932                       # margins, reused by every helper

def new_page():
    return plt.figure(figsize=(8.5, 11), facecolor="white")

def hrule(fig, y, c=RULE, lw=0.9):
    fig.add_artist(plt.Line2D([L, R], [y, y], color=c, lw=lw, transform=fig.transFigure))

def para(fig, y, text, size=9.5, width=112, lead=0.0155):
    for line in textwrap.wrap(text, width):
        fig.text(L, y, line, fontsize=size, color=BODY, va="top"); y -= lead
    return y
```

Write five such helpers once — `page_head`, `para`, `sec`, `table`, `stat_row` — and every
page after the first costs minutes. Have each **return the new `y`** so pages compose by
chaining. Draw tables as positioned text with hairline rules between rows; ASCII tables in
a monospace block are the main thing that makes a PDF look like a terminal dump.

Add quiet page furniture — a footer with the project name and a page number in `MUTED` at
7.6pt. Cheap, and it makes a loose printed page identifiable.

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
with PdfPages(path) as pdf:
    for build in (page_groups, page_issues, page_method):
        build(pdf)                      # each ends in pdf.savefig(fig); plt.close(fig)
```

Nice fonts have small character sets, and matplotlib **silently** substitutes a missing
glyph into a visible box. Arrows are the usual casualty: `→` and `←` are absent from both
Avenir Next and Helvetica Neue, while `≤ × · − ≈ ² ½` are generally fine. Audit the whole
document rather than eyeballing it:

```python
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    build_report(path)
print(sorted({str(x.message) for x in w if "Glyph" in str(x.message)}) or "clean")
```

Fix by substituting ASCII (`->`), not by abandoning the font.

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
- `pdf.fonttype = 42`, a discovered font, softened ink, hairline rules.
- Figure titles state conclusions and are left-aligned to the text column.
- Glyph warnings audited to zero.
- Every metric and flag defined inside the document.
- Exemplars include a failure, not only successes.
