"""Interactive PK profile explorer: hover a curve, see the compound."""

import logging

import numpy as np
import pandas as pd

from workbench.utils.apps._serve import serve
from workbench.utils.chem_utils.vis import svg_from_smiles
from workbench.utils.plots.pk import _concentration, _tmax

log = logging.getLogger("workbench")

# mL/min/kg -> L/h/kg. CL and Vd are reported in units that do not agree.
CL_TO_L_PER_H = 0.06

BACKGROUND = "rgba(24, 24, 24, 1)"
TEXT_COLOR = "#e0e0e0"
GROUP_COLORS = ["#4c9be8", "#e8804c", "#5fd08a", "#c77dd6", "#e8c84c", "#e85f7d"]


def derive(df: pd.DataFrame, cl_col: str, vd_col: str, dose: float = 1.0) -> pd.DataFrame:
    """Add `ke`, `t_half`, and `auc` from clearance and volume of distribution.

    Args:
        df (pd.DataFrame): One row per compound.
        cl_col (str): Clearance column, mL/min/kg.
        vd_col (str): Volume of distribution column, L/kg.
        dose (float): Dose in mg/kg, for AUC. Defaults to 1.0.

    Returns:
        pd.DataFrame: Copy with ke (1/h), t_half (h), and auc added.
    """
    out = df.copy()
    out["ke"] = CL_TO_L_PER_H * out[cl_col] / out[vd_col]
    out["t_half"] = np.log(2) / out["ke"]
    out["auc"] = dose / (CL_TO_L_PER_H * out[cl_col])
    return out


def cluster_pk_plane(df: pd.DataFrame, cl_col: str, vd_col: str, n_clusters: int = 4) -> pd.Series:
    """Group compounds by where they sit on the (log CL, log Vd) plane.

    The plane separates compounds that profile space cannot: two groups with the same
    half-life still differ in volume of distribution. Labels are renumbered so group 0
    is the slowest by median half-life.

    Args:
        df (pd.DataFrame): Frame carrying `t_half`, from :func:`derive`.
        cl_col (str): Clearance column.
        vd_col (str): Volume of distribution column.
        n_clusters (int): How many groups. Defaults to 4.

    Returns:
        pd.Series: Integer group label per row, ordered slow to fast.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    plane = np.log10(df[[cl_col, vd_col]].to_numpy())
    raw = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(plane))

    # Slowest first, so group color tracks duration rather than k-means' arbitrary order.
    order = pd.Series(df["t_half"].to_numpy()).groupby(raw).median().sort_values(ascending=False).index
    return pd.Series(raw, index=df.index).map({old: new for new, old in enumerate(order)})


def _curves(df: pd.DataFrame, vd_col: str, dose: float, f_percent: float, ka: float, duration: float) -> tuple:
    """Time grid, one concentration curve per row, and each row's (tmax, Cmax)."""
    t = np.linspace(0, duration, 300)
    f_dose = dose * f_percent / 100.0

    curves, peaks = [], []
    for row in df.itertuples():
        ke, vd = row.ke, getattr(row, vd_col)
        curves.append(_concentration(t, f_dose, vd, ke, ka))
        tmax = _tmax(ke, ka)
        peaks.append((tmax, float(_concentration(np.array([tmax]), f_dose, vd, ke, ka)[0])))
    return t, curves, peaks


def _build_figure(df, vd_col, id_col, group_label, dose, f_percent, ka, duration, y_floor):
    """The figure, plus a trace-index -> compound-id list for resolving hover.

    `trace_ids` runs parallel to `fig.data`: a hover event gives a curve number, and
    this turns it into a compound. Marker traces carry `customdata` instead and hold
    "" here, so the two lookups never disagree.
    """
    import plotly.graph_objects as go

    t, curves, peaks = _curves(df, vd_col, dose, f_percent, ka, duration)
    order = sorted(df["pk_group"].unique())
    colors = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(order)}

    fig = go.Figure()
    trace_ids = []
    seen = set()
    for i, row in enumerate(df.itertuples()):
        group = row.pk_group
        fig.add_trace(
            go.Scatter(
                x=t,
                y=curves[i],
                mode="lines",
                line=dict(color=colors[group], width=1.5),
                name=str(group),
                legendgroup=str(group),
                legendrank=order.index(group),  # legend follows group order, not the frame's row order
                showlegend=group not in seen,  # one legend entry per group toggles all its curves
                hoverinfo="skip",
            )
        )
        trace_ids.append(getattr(row, id_col))
        seen.add(group)

    # Peaks as their own traces: a marker is a far bigger hover target than a thin line.
    for group in order:
        rows = [i for i, row in enumerate(df.itertuples()) if row.pk_group == group]
        fig.add_trace(
            go.Scatter(
                x=[peaks[i][0] for i in rows],
                y=[peaks[i][1] for i in rows],
                mode="markers",
                marker=dict(color=colors[group], size=9, line=dict(color="#181818", width=1)),
                name=str(group),
                legendgroup=str(group),
                showlegend=False,
                customdata=[trace_ids[i] for i in rows],
                hovertemplate="%{customdata}<br>tmax %{x:.2f} h<br>Cmax %{y:.3g}<extra></extra>",
            )
        )
        trace_ids.append("")

    peak_max = max(p[1] for p in peaks)
    floor = y_floor if y_floor is not None else peak_max * 1e-3
    fig.update_layout(
        title=dict(
            text=(
                f"Oral PK profiles, n={len(df)} — grouped by {group_label}<br>"
                f"<sub>assumes ka = {ka:g} /h and F = {f_percent:g}%, neither in the data; "
                f"dose {dose:g} mg/kg</sub>"
            ),
            font=dict(size=15),
        ),
        xaxis=dict(title="Time (h)", gridcolor="#333"),
        yaxis=dict(
            title="Concentration (mg/L)",
            type="log",
            gridcolor="#333",
            range=[np.log10(floor), np.log10(peak_max * 2)],
        ),
        hovermode="closest",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR),
        legend=dict(title=group_label),
        margin=dict(t=80),
        height=720,
    )
    return fig, trace_ids, colors


def build_app(
    df: pd.DataFrame,
    cl_col: str = "cl",
    vd_col: str = "vd",
    smiles_col: str = "smiles",
    id_col: str = "id",
    group_col: str = None,
    n_clusters: int = 4,
    half_life_col: str = None,
    dose: float = 1.0,
    ka: float = 1.0,
    f_percent: float = 100.0,
    duration: float = 24.0,
    y_floor: float = None,
):
    """Build the hoverable PK profile app: one curve per compound, structure on hover.

    Curves are the oral Bateman profile. **`ka` and `f_percent` are assumptions, not
    data** — an IV frame carries neither, and both are stated in the plot title because
    moving `ka` slides every tmax. AUC does not move with either.

    Rows missing clearance or volume of distribution are dropped, and the count is
    logged; a silent drop would misrepresent how much of the set is on screen.

    Args:
        df (pd.DataFrame): One row per compound.
        cl_col (str): Clearance column, mL/min/kg. Defaults to "cl".
        vd_col (str): Volume of distribution column, L/kg. Defaults to "vd".
        smiles_col (str): SMILES column, for the structure panel. Defaults to "smiles".
        id_col (str): Compound id column. Defaults to "id".
        group_col (str, optional): Column to color and group by. Clusters on the
            (log CL, log Vd) plane when omitted.
        n_clusters (int): Groups to cluster into, when `group_col` is omitted.
        half_life_col (str, optional): Reported half-life, cross-checked against the
            derived one. Correlation is logged; below 0.95 means the units disagree.
        dose (float): mg/kg. Defaults to 1.0.
        ka (float): Absorption rate constant, 1/h. Defaults to 1.0.
        f_percent (float): Assumed bioavailability. Defaults to 100.0.
        duration (float): Hours to plot. Defaults to 24.0.
        y_floor (float, optional): Lower bound of the log y-axis. Defaults to three
            decades below the highest peak — concentrations far below any assay's limit
            of quantitation are arithmetic, not measurement. Pass a real LLOQ if known.

    Returns:
        dash.Dash: The built app, callbacks registered. Hand it to
            :func:`workbench.utils.apps._serve.serve`, or edit its layout first —
            `pk_explorer` is the one-call version.
    """
    from dash import Dash, Input, Output, dcc, html

    missing = [c for c in (cl_col, vd_col, smiles_col, id_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not in the frame: {missing}. Have: {list(df.columns)}")

    usable = df.dropna(subset=[cl_col, vd_col]).copy()
    dropped = len(df) - len(usable)
    if dropped:
        log.important(f"Dropped {dropped} of {len(df)} rows with no {cl_col} or {vd_col}")
    if usable.empty:
        raise ValueError(f"No rows with both {cl_col} and {vd_col}")
    if (usable[[cl_col, vd_col]] <= 0).any().any():
        raise ValueError(f"{cl_col} and {vd_col} must be positive; a log plane and a rate need them")

    usable = derive(usable, cl_col, vd_col, dose)

    if half_life_col and half_life_col in usable.columns:
        pair = usable[["t_half", half_life_col]].dropna()
        corr = pair["t_half"].corr(pair[half_life_col]) if len(pair) > 1 else float("nan")
        log.important(f"Derived t½ vs {half_life_col}: corr {corr:.3f} (n={len(pair)})")
        if corr < 0.95:
            log.warning(f"Derived and reported half-life disagree (corr {corr:.3f}) — check units")

    if group_col:
        groups = usable[group_col].astype(str)
        group_label = group_col
    else:
        groups = cluster_pk_plane(usable, cl_col, vd_col, n_clusters).map(lambda k: f"C{k}")
        group_label = "cluster"
    usable["pk_group"] = groups.to_numpy()

    fig, trace_ids, colors = _build_figure(usable, vd_col, id_col, group_label, dose, f_percent, ka, duration, y_floor)

    app = Dash(__name__)
    panel = {"padding": "12px", "color": TEXT_COLOR, "fontFamily": "system-ui, sans-serif"}
    app.layout = html.Div(
        style={"display": "flex", "backgroundColor": BACKGROUND, "minHeight": "100vh"},
        children=[
            html.Div(dcc.Graph(id="pk-graph", figure=fig, style={"height": "740px"}), style={"width": "68%"}),
            html.Div(
                style={"width": "32%", **panel},
                children=[
                    html.H3("Hover a curve", id="pk-title", style={"marginTop": "8px"}),
                    html.Img(id="pk-mol", style={"width": "100%", "maxWidth": "420px"}),
                    html.Div(id="pk-stats", style={"fontSize": "14px", "lineHeight": "1.7"}),
                ],
            ),
        ],
    )

    lookup = usable.set_index(id_col)

    @app.callback(
        # Writes the three leaves, never the Div holding them -- replacing that container
        # swaps in id-less copies and every later hover fires at outputs that are gone.
        Output("pk-title", "children"),
        Output("pk-title", "style"),
        Output("pk-mol", "src"),
        Output("pk-stats", "children"),
        Input("pk-graph", "hoverData"),
    )
    def _on_hover(hover):
        if not hover:
            return "Hover a curve", {"marginTop": "8px"}, None, None
        point = hover["points"][0]
        mol_id = point.get("customdata") or trace_ids[point["curveNumber"]]
        if not mol_id or mol_id not in lookup.index:
            return "Hover a curve", {"marginTop": "8px"}, None, None

        row = lookup.loc[mol_id]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        color = colors[row["pk_group"]]
        image = svg_from_smiles(row[smiles_col], width=420, height=340, background=BACKGROUND)

        stats = [(group_label, row["pk_group"]), (cl_col, f"{row[cl_col]:.3g}"), (vd_col, f"{row[vd_col]:.3g}")]
        stats += [("t½ (derived)", f"{row['t_half']:.2f} h"), ("AUC", f"{row['auc']:.3g}")]
        if half_life_col and half_life_col in lookup.columns and pd.notna(row[half_life_col]):
            stats.append(("t½ (reported)", f"{row[half_life_col]:.2f} h"))

        table = html.Table(
            [html.Tr([html.Td(k, style={"paddingRight": "16px", "opacity": 0.7}), html.Td(v)]) for k, v in stats]
        )
        return str(mol_id), {"marginTop": "8px", "color": color}, image, table

    return app


def pk_explorer(df: pd.DataFrame, port: int = None, open_browser: bool = True, **kwargs) -> str:
    """Build the PK explorer and serve it. See :func:`build_app` for the arguments.

    Args:
        df (pd.DataFrame): One row per compound.
        port (int, optional): Port to serve on. Defaults to the first free one.
        open_browser (bool): Open a browser tab. Defaults to True.
        **kwargs: Passed through to :func:`build_app`.

    Returns:
        str: The URL the app is serving on.
    """
    return serve(build_app(df, **kwargs), port=port, open_browser=open_browser)
