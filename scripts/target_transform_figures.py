"""Generate the target-transform figures in docs/models/target_transforms.md.

Three SVGs, all measured on real OpenADMET assay data rather than synthetic curves:

    target_transform_domains.svg      each transform's forward map and its valid domain
    target_transform_resolution.svg   where each transform spends its output range
    target_transform_ceiling.svg      what a logit's ceiling trades between the tails

Run it from the repo root after changing a transform or its defaults:

    python scripts/target_transform_figures.py

Styling matches the other docs SVGs: transparent background, light text for the dark
docs theme, and a validated three-hue categorical palette.
"""

from pathlib import Path

import numpy as np

from workbench.api import PublicData
from workbench.utils.target_transform import ArcSinh, Log10, Logit

OUT_DIR = Path("docs/images")

# Dark-surface categorical slots 1-3, which validate on every pair, plus recessive ink
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
BASELINE = "#8a8a84"
TITLE, LABEL, AXIS, GRID, BAD = "#ccc", "#aaa", "#999", "#3a3a38", "#e66767"

FONT = 'font-family="Helvetica, Arial"'

ASSAYS = {
    "ksol": "comp_chem/openadmet/expansionrx/train/ksol",
    "mppb": "comp_chem/openadmet/expansionrx/train/mppb",
    "mbpb": "comp_chem/openadmet/expansionrx/train/mbpb",
    "mlm_clint": "comp_chem/openadmet/expansionrx/train/mlm_clint",
    "logd": "comp_chem/openadmet/expansionrx/train/logd",
}


def load_assays() -> dict:
    """Each assay's measured target values, NaNs dropped."""
    public_data = PublicData()
    values = {}
    for column, path in ASSAYS.items():
        series = public_data.get(path)[column].dropna()
        values[column] = series.values.astype(float)
        print(f"{column:10s} n={len(series):5d} min={values[column].min():g} max={values[column].max():g}")
    return values


def decile_shares(transform, y: np.ndarray) -> tuple:
    """Share of the transform's output range held by the bottom and top decile of the data.

    A linear map gives each decile 10%. More than that is resolution the transform has
    moved to that tail, which is the whole reason to reach for one.
    """
    quantiles = np.quantile(y, [0.0, 0.1, 0.9, 1.0])
    low, q10, q90, high = transform.forward(quantiles)
    full = high - low
    return (q10 - low) / full, (high - q90) / full


# --------------------------------------------------------------------------------------
# SVG primitives
# --------------------------------------------------------------------------------------
def svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n{body}\n</svg>\n'
    )


def text(x, y, content, size=16, fill=LABEL, anchor="start", weight="normal", extra="") -> str:
    content = str(content).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" {FONT} '
        f'font-size="{size}" font-weight="{weight}"{extra}>{content}</text>'
    )


def line(x1, y1, x2, y2, stroke=AXIS, width=1.5, dash="") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def rect(x, y, w, h, fill, opacity=1.0, rx=0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" opacity="{opacity}"/>'
    )


def polyline(points, stroke, width=2.5) -> str:
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>'


class Scale:
    """Maps data coordinates onto a pixel box."""

    def __init__(self, x_range, y_range, box):
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.left, self.top, self.width, self.height = box

    def x(self, value):
        return self.left + (value - self.x0) / (self.x1 - self.x0) * self.width

    def y(self, value):
        return self.top + self.height - (value - self.y0) / (self.y1 - self.y0) * self.height


# --------------------------------------------------------------------------------------
# Figure 1: forward maps and valid domains
# --------------------------------------------------------------------------------------
def domains_figure(path: Path) -> None:
    """Three panels: the shape of each map, with the region it refuses marked."""
    width, panel_h, gap = 860, 215, 24
    left, plot_w = 118, 600
    height = 70 + 3 * (panel_h + gap)

    panels = [
        {
            "title": "log10(y + offset)",
            "note": "valid where y + offset > 0 — an offset lifts the zeros",
            "colour": BLUE,
            "transform": Log10(offset=0.1),
            "x_range": (-0.6, 10.0),
            "invalid": [(-0.6, -0.1)],
            "ticks": [0, 2, 4, 6, 8, 10],
        },
        {
            "title": "log10(y / (ceiling - y))      ceiling = 120",
            "note": "valid where 0 < y < ceiling — both ends are refused",
            "colour": AQUA,
            "transform": Logit(ceiling=120.0),
            "x_range": (-8.0, 132.0),
            "invalid": [(-8.0, 0.0), (120.0, 132.0)],
            "ticks": [0, 30, 60, 90, 120],
        },
        {
            "title": "asinh(y / scale)      scale = 1",
            "note": "valid everywhere — zeros and negatives pass straight through",
            "colour": ORANGE,
            "transform": ArcSinh(scale=1.0),
            "x_range": (-10.0, 10.0),
            "invalid": [],
            "ticks": [-10, -5, 0, 5, 10],
        },
    ]

    body = [
        text(width / 2, 36, "What each transform does, and where it is defined", 24, TITLE, "middle", "bold"),
        rect(width / 2 - 96, 52, 20, 12, BAD, 0.3, 2),
        text(width / 2 - 68, 62, "outside the domain", 14, LABEL),
    ]

    for index, panel in enumerate(panels):
        top = 70 + index * (panel_h + gap)
        plot_top, plot_h = top + 42, panel_h - 78
        x_lo, x_hi = panel["x_range"]

        # Sample only the valid side of the domain
        xs = np.linspace(x_lo, x_hi, 600)
        valid = np.ones(xs.shape, dtype=bool)
        for bad_lo, bad_hi in panel["invalid"]:
            valid &= ~((xs >= bad_lo) & (xs <= bad_hi))
        xs_valid = xs[valid]
        ys_valid = panel["transform"].forward(xs_valid)
        finite = np.isfinite(ys_valid)
        xs_valid, ys_valid = xs_valid[finite], ys_valid[finite]

        # Clip the asymptotes so one exploding end does not flatten the rest
        y_lo, y_hi = np.quantile(ys_valid, [0.02, 0.98])
        pad = 0.18 * (y_hi - y_lo)
        y_lo, y_hi = y_lo - pad, y_hi + pad
        scale = Scale((x_lo, x_hi), (y_lo, y_hi), (left, plot_top, plot_w, plot_h))

        body.append(text(left, top + 24, panel["title"], 19, TITLE, "start", "bold"))
        body.append(text(left + plot_w + 16, plot_top + 16, "output", 14, LABEL))

        # Invalid regions, marked before the curve so the curve sits on top
        for bad_lo, bad_hi in panel["invalid"]:
            x_start, x_end = scale.x(max(bad_lo, x_lo)), scale.x(min(bad_hi, x_hi))
            body.append(rect(x_start, plot_top, x_end - x_start, plot_h, BAD, 0.14))
            body.append(line(x_end, plot_top, x_end, plot_top + plot_h, BAD, 1.5, "5 4"))

        # Zero line, axes, ticks
        if y_lo < 0 < y_hi:
            body.append(line(left, scale.y(0), left + plot_w, scale.y(0), GRID, 1.2))
        body.append(line(left, plot_top, left, plot_top + plot_h, AXIS, 1.5))
        body.append(line(left, plot_top + plot_h, left + plot_w, plot_top + plot_h, AXIS, 1.5))
        for tick in panel["ticks"]:
            tick_x = scale.x(tick)
            body.append(line(tick_x, plot_top + plot_h, tick_x, plot_top + plot_h + 5, AXIS, 1.2))
            body.append(text(tick_x, plot_top + plot_h + 23, f"{tick:g}", 14, LABEL, "middle"))
        for value in (y_lo + pad, y_hi - pad):
            body.append(text(left - 10, scale.y(value) + 5, f"{value:.1f}", 14, LABEL, "end"))

        # Clipped: a logit runs to infinity at both ends, and only the plot box is in view
        curve = polyline([(scale.x(x), scale.y(y)) for x, y in zip(xs_valid, ys_valid)], panel["colour"], 3)
        body.append(f'<clipPath id="panel{index}"><rect x="{left}" y="{plot_top}" ')
        body[-1] += f'width="{plot_w}" height="{plot_h}"/></clipPath>'
        body.append(f'<g clip-path="url(#panel{index})">{curve}</g>')
        body.append(text(left + plot_w / 2, plot_top + plot_h + 46, panel["note"], 15, LABEL, "middle"))

    path.write_text(svg(width, height, "\n".join(body)), encoding="utf-8")
    print(f"wrote {path}")


# --------------------------------------------------------------------------------------
# Figure 2: where the output range goes, on real assay distributions
# --------------------------------------------------------------------------------------
def resolution_figure(path: Path, assays: dict) -> None:
    """Bottom-decile share of the output range, per assay, per candidate transform."""
    clint = assays["mlm_clint"]
    clint_floor = float(clint[clint > 0].min())
    mppb_floored = np.clip(assays["mppb"], 0.1, None)

    groups = [
        (
            "ksol — kinetic solubility, 5 decades, no zeros",
            [("linear", BASELINE, None), ("log10", BLUE, Log10())],
            assays["ksol"],
            None,
        ),
        (
            "mppb — percent unbound, bounded [0, 100], 10 zeros",
            [
                ("linear", BASELINE, None),
                ("log10 offset 0.1", BLUE, Log10(offset=0.1)),
                ("logit ceiling 120", AQUA, Logit(ceiling=120.0)),
            ],
            assays["mppb"],
            mppb_floored,
        ),
        (
            f"mlm_clint — 5 decades and 147 exact zeros (floor {clint_floor:g})",
            [
                ("linear", BASELINE, None),
                (f"log10 offset {clint_floor:g}", BLUE, Log10(offset=clint_floor)),
                (f"arcsinh scale {clint_floor:g}", ORANGE, ArcSinh(scale=clint_floor)),
            ],
            clint,
            None,
        ),
    ]

    # A two-sided bar: the low tail grows left of centre, the high tail right, so a
    # transform that buys one tail by spending the other cannot hide it
    width, centre, half, max_share = 880, 548, 268, 0.62
    name_x, row_h, bar_h, group_gap = 236, 32, 20, 26
    y = 128
    body = [
        text(width / 2, 36, "Where each transform spends its output range", 24, TITLE, "middle", "bold"),
        text(
            width / 2,
            62,
            "share of the output range held by the lowest and highest 10% of the measured values",
            15,
            LABEL,
            "middle",
        ),
        text(centre - 14, 96, "◀ lowest 10%", 15, LABEL, "end"),
        text(centre + 14, 96, "highest 10% ▶", 15, LABEL),
    ]

    def bar_px(share):
        return half * min(share, max_share) / max_share

    for title, candidates, y_values, floored in groups:
        body.append(text(26, y + 2, title, 16, TITLE, "start", "bold"))
        y += 24
        for name, colour, transform in candidates:
            data = floored if isinstance(transform, Logit) else y_values
            low, high = (0.1, 0.1) if transform is None else decile_shares(transform, data)
            low_px, high_px = bar_px(low), bar_px(high)
            body.append(text(name_x, y + bar_h - 5, name, 15, LABEL, "end"))
            body.append(rect(centre - low_px, y, low_px - 1, bar_h, colour, 1.0, 4))
            body.append(rect(centre + 1, y, high_px - 1, bar_h, colour, 1.0, 4))
            body.append(text(centre - low_px - 10, y + bar_h - 5, f"{100 * low:.0f}%", 15, TITLE, "end"))
            body.append(text(centre + high_px + 10, y + bar_h - 5, f"{100 * high:.0f}%", 15, TITLE))
            y += row_h
        y += group_gap

    # The linear reference on both sides, and the centre these bars grow from
    bottom = y - group_gap + 4
    for side in (-1, 1):
        reference_x = centre + side * bar_px(0.1)
        body.insert(4, line(reference_x, 104, reference_x, bottom, LABEL, 1.2, "4 4"))
    body.insert(4, line(centre, 104, centre, bottom, AXIS, 1.5))
    body.append(text(centre, bottom + 26, "dashed: the 10% a linear target gives each tail", 14, LABEL, "middle"))

    path.write_text(svg(width, int(bottom + 44), "\n".join(body)), encoding="utf-8")
    print(f"wrote {path}")


# --------------------------------------------------------------------------------------
# Figure 3: the logit ceiling trade-off, on an assay that reaches its bound
# --------------------------------------------------------------------------------------
def ceiling_figure(path: Path, assays: dict) -> None:
    """mbpb reaches 99.23 percent bound, so the ceiling choice is visible in the data."""
    mbpb = np.clip(assays["mbpb"], 0.1, None)
    observed_max = float(mbpb.max())

    width, height = 860, 470
    left, top, plot_w, plot_h = 96, 108, 560, 250
    ceilings = [(100.5, BLUE), (105.0, AQUA), (120.0, ORANGE)]

    body = [
        text(width / 2, 40, "What a logit ceiling trades between the tails", 24, TITLE, "middle", "bold"),
        text(
            width / 2,
            68,
            f"mbpb percent bound, {len(mbpb)} rows reaching {observed_max:g} — each curve scaled to its own range",
            15,
            LABEL,
            "middle",
        ),
    ]

    scale = Scale((0, 100), (0, 1), (left, top, plot_w, plot_h))
    body.append(line(left, top, left, top + plot_h, AXIS, 1.5))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, AXIS, 1.5))
    for tick in (0, 25, 50, 75, 100):
        body.append(text(scale.x(tick), top + plot_h + 23, f"{tick:g}", 14, LABEL, "middle"))
    for tick in (0.0, 0.5, 1.0):
        body.append(line(left, scale.y(tick), left + plot_w, scale.y(tick), GRID, 1.0))
        body.append(text(left - 10, scale.y(tick) + 5, f"{tick:g}", 14, LABEL, "end"))
    body.append(text(left + plot_w / 2, top + plot_h + 48, "percent bound", 15, LABEL, "middle"))

    xs = np.linspace(0.1, observed_max, 400)
    legend_y = top + 18
    for ceiling, colour in ceilings:
        transform = Logit(ceiling=ceiling)
        forward = transform.forward(xs)
        normalized = (forward - forward.min()) / (forward.max() - forward.min())
        body.append(polyline([(scale.x(x), scale.y(v)) for x, v in zip(xs, normalized)], colour, 3))

        low, high = decile_shares(transform, mbpb)
        body.append(rect(left + plot_w + 36, legend_y - 11, 26, 4, colour, 1.0, 2))
        body.append(text(left + plot_w + 72, legend_y - 4, f"ceiling {ceiling:g}", 15, TITLE))
        body.append(text(left + plot_w + 72, legend_y + 18, f"tails {100 * low:.0f}% / {100 * high:.0f}%", 14, LABEL))
        legend_y += 58

    body.append(text(left + plot_w + 36, legend_y + 4, "bottom / top decile", 14, LABEL))
    body.append(
        text(
            left,
            height - 34,
            "A tight ceiling buys top-tail resolution, and turns a value near the bound into an outlier.",
            15,
            LABEL,
        )
    )
    body.append(text(left, height - 12, "A ceiling at the physical bound refuses any row that reaches it.", 15, LABEL))

    path.write_text(svg(width, height, "\n".join(body)), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    assays = load_assays()
    print()
    domains_figure(OUT_DIR / "target_transform_domains.svg")
    resolution_figure(OUT_DIR / "target_transform_resolution.svg", assays)
    ceiling_figure(OUT_DIR / "target_transform_ceiling.svg", assays)

    print("\nMeasured shares (bottom / top decile of the output range):")
    clint = assays["mlm_clint"]
    clint_floor = float(clint[clint > 0].min())
    rows = [
        ("ksol", "log10", Log10(), assays["ksol"]),
        ("mppb", "log10 offset 0.1", Log10(offset=0.1), assays["mppb"]),
        ("mppb", "logit ceiling 120", Logit(ceiling=120.0), np.clip(assays["mppb"], 0.1, None)),
        ("mppb", "logit ceiling 100.5", Logit(ceiling=100.5), np.clip(assays["mppb"], 0.1, None)),
        ("mbpb", "logit ceiling 120", Logit(ceiling=120.0), np.clip(assays["mbpb"], 0.1, None)),
        ("mlm_clint", f"log10 offset {clint_floor:g}", Log10(offset=clint_floor), clint),
        ("mlm_clint", f"arcsinh scale {clint_floor:g}", ArcSinh(scale=clint_floor), clint),
        ("logd", "arcsinh scale 1", ArcSinh(scale=1.0), assays["logd"]),
    ]
    for assay, name, transform, data in rows:
        low, high = decile_shares(transform, data)
        print(f"  {assay:10s} {name:22s} {100 * low:5.1f}% / {100 * high:5.1f}%")


if __name__ == "__main__":
    main()
