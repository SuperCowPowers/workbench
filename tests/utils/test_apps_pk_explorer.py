"""Tests for the interactive PK explorer."""

import numpy as np
import pandas as pd
import pytest

from workbench.utils.apps.pk_explorer import _build_figure, build_app, cluster_pk_plane, derive

SMILES = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccccc1"]


def frame(n_slow=6, n_fast=18, seed=0):
    """Bimodal PK plane: a low-clearance island and a high-clearance cloud."""
    rng = np.random.default_rng(seed)
    cl = np.concatenate([10 ** rng.normal(-1.4, 0.3, n_slow), 10 ** rng.normal(1.6, 0.35, n_fast)])
    vd = np.concatenate([10 ** rng.normal(-0.85, 0.15, n_slow), 10 ** rng.normal(0.4, 0.25, n_fast)])
    n = n_slow + n_fast
    df = pd.DataFrame(
        {
            "id": [f"IDC-{20000 + i}" for i in range(n)],
            "smiles": [SMILES[i % len(SMILES)] for i in range(n)],
            "cl": cl,
            "vd": vd,
        }
    )
    df["half_life_hr"] = np.log(2) / (0.06 * df["cl"] / df["vd"])
    return df


def grouped(df=None):
    """A derived frame with a `pk_group` column, ready for _build_figure."""
    df = frame() if df is None else df
    out = derive(df, "cl", "vd")
    out["pk_group"] = cluster_pk_plane(out, "cl", "vd", 4).map(lambda k: f"C{k}").to_numpy()
    return out


def test_derive_matches_the_closed_forms():
    """ke, t half, and AUC each have one right answer given CL and Vd."""
    df = derive(pd.DataFrame({"cl": [10.0], "vd": [2.0]}), "cl", "vd", dose=1.0)
    assert df["ke"].iloc[0] == pytest.approx(0.06 * 10.0 / 2.0)
    assert df["t_half"].iloc[0] == pytest.approx(np.log(2) / 0.3)
    assert df["auc"].iloc[0] == pytest.approx(1.0 / 0.6)


def test_derived_half_life_reproduces_a_reported_column():
    """The unit conversion is the trap; a reported half-life is the check on it."""
    df = derive(frame(), "cl", "vd")
    assert df["t_half"].corr(df["half_life_hr"]) == pytest.approx(1.0, abs=1e-6)


def test_clusters_are_ordered_slow_to_fast():
    """Group 0 must be the longest-lived, so color tracks duration not k-means order."""
    df = derive(frame(), "cl", "vd")
    labels = cluster_pk_plane(df, "cl", "vd", 4)
    medians = df.groupby(labels.to_numpy())["t_half"].median()
    assert list(medians.index) == sorted(medians.index)
    assert medians.is_monotonic_decreasing


def test_trace_ids_run_parallel_to_traces():
    """Hover resolves by curve number, so a misaligned list names the wrong compound."""
    df = grouped()
    fig, trace_ids, _ = _build_figure(df, "vd", "id", "cluster", 1.0, 100.0, 1.0, 24.0, None)
    assert len(trace_ids) == len(fig.data)

    line_ids = [tid for tid, tr in zip(trace_ids, fig.data) if tr.mode == "lines"]
    assert line_ids == list(df["id"])
    assert all(tid == "" for tid, tr in zip(trace_ids, fig.data) if tr.mode == "markers")


def test_marker_traces_carry_their_ids_as_customdata():
    """Markers are the big hover target; they resolve via customdata, not the list."""
    df = grouped()
    fig, trace_ids, _ = _build_figure(df, "vd", "id", "cluster", 1.0, 100.0, 1.0, 24.0, None)
    from_markers = [cd for tr in fig.data if tr.mode == "markers" for cd in tr.customdata]
    assert sorted(from_markers) == sorted(df["id"])


def test_one_legend_entry_perpk_group():
    """A legend click toggles a whole cluster only if exactly one member carries it."""
    df = grouped()
    fig, _, _ = _build_figure(df, "vd", "id", "cluster", 1.0, 100.0, 1.0, 24.0, None)
    shown = [tr.legendgroup for tr in fig.data if tr.showlegend]
    assert sorted(shown) == sorted(df["pk_group"].unique())


def test_curves_share_one_color_perpk_group():
    df = grouped()
    fig, _, colors = _build_figure(df, "vd", "id", "cluster", 1.0, 100.0, 1.0, 24.0, None)
    for trace in fig.data:
        expected = colors[trace.legendgroup]
        assert (trace.line.color if trace.mode == "lines" else trace.marker.color) == expected


def test_y_axis_floors_instead_of_plotting_fiction():
    """Bateman decays forever; three decades below the top peak is the honest window."""
    df = grouped()
    fig, _, _ = _build_figure(df, "vd", "id", "cluster", 1.0, 100.0, 1.0, 24.0, None)
    low, high = fig.layout.yaxis.range
    assert high - low == pytest.approx(np.log10(2e3), abs=0.01)

    fig, _, _ = _build_figure(df, "vd", "id", "cluster", 1.0, 100.0, 1.0, 24.0, 1e-2)
    assert fig.layout.yaxis.range[0] == pytest.approx(np.log10(1e-2))


def test_title_states_the_invented_assumptions():
    """ka and F are not in the data; a plot that hides that is misleading."""
    df = grouped()
    fig, _, _ = _build_figure(df, "vd", "id", "cluster", 1.0, 100.0, 1.0, 24.0, None)
    assert "ka = 1 /h" in fig.layout.title.text
    assert "F = 100%" in fig.layout.title.text


def test_missing_columns_name_what_is_there():
    with pytest.raises(ValueError, match="Columns not in the frame"):
        build_app(frame().drop(columns=["smiles"]))


def test_non_positive_inputs_are_rejected():
    df = frame()
    df.loc[0, "cl"] = 0.0
    with pytest.raises(ValueError, match="must be positive"):
        build_app(df)


def test_all_rows_missing_clearance_raises():
    df = frame()
    df["cl"] = np.nan
    with pytest.raises(ValueError, match="No rows with both"):
        build_app(df)


def _hover_handler(app):
    """The undecorated callback -- Dash's wrapper needs a live request context."""
    return next(iter(app.callback_map.values()))["callback"].__wrapped__


def test_callback_writes_only_leaf_components():
    """Writing the container that holds the other outputs replaces them with id-less
    copies, and every hover after the first fires at components that no longer exist.
    The panel updates once, then silently stops."""
    app = build_app(frame())
    outputs = {o.component_id for entry in app.callback_map.values() for o in entry["output"]}
    assert outputs == {"pk-title", "pk-mol", "pk-stats"}


def test_hover_resolves_to_a_compound():
    """The whole point of the app: a hover event names the right molecule."""
    handler = _hover_handler(build_app(frame()))

    empty_title, _, empty_img, empty_stats = handler(None)
    assert empty_img is None and empty_stats is None and "Hover" in empty_title

    df = grouped()
    fig, trace_ids, _ = _build_figure(df, "vd", "id", "cluster", 1.0, 100.0, 1.0, 24.0, None)
    title, style, img, stats = handler({"points": [{"curveNumber": 3}]})
    assert title == trace_ids[3]
    assert img.startswith("data:image/svg+xml;base64,")
    assert style["color"].startswith("#")
    assert stats is not None


def test_hover_on_a_marker_uses_customdata():
    """Marker traces hold "" in trace_ids, so they must resolve via customdata."""
    handler = _hover_handler(build_app(frame()))
    title, _, img, _ = handler({"points": [{"curveNumber": 0, "customdata": "IDC-20005"}]})
    assert title == "IDC-20005"
    assert img.startswith("data:image/svg+xml;base64,")


def test_legend_follows_group_order():
    """Entries appear in trace order by default, which scrambles them against the groups."""
    df = grouped()
    fig, _, _ = _build_figure(df, "vd", "id", "cluster", 1.0, 100.0, 1.0, 24.0, None)
    shown = [tr for tr in fig.data if tr.showlegend]
    assert [tr.legendgroup for tr in sorted(shown, key=lambda tr: tr.legendrank)] == sorted(df["pk_group"].unique())


def test_serve_quiets_the_per_request_loggers():
    """A hover is one request; werkzeug logging each at INFO buries the REPL."""
    import logging

    from workbench.utils.apps._serve import serve

    for name in ("werkzeug", "dash", "dash.dash"):
        logging.getLogger(name).setLevel(logging.INFO)

    serve(build_app(frame()), open_browser=False)
    for name in ("werkzeug", "dash", "dash.dash"):
        assert logging.getLogger(name).level == logging.WARNING
