"""Plots for compound neighborhoods."""

from workbench.utils.chem_utils.vis import img_from_smiles


def graph(
    query_id,
    neighbors_df,
    target_col,
    smiles_col: str = "smiles",
    id_col: str = "neighbor_id",
    similarity_col: str = "similarity",
    n_neighbors: int = 5,
    cmap_name: str = "viridis",
    title: str = None,
):
    """Radial graph of a query compound and its nearest neighbors.

    The query sits at the center with the `n_neighbors` most-similar neighbors around
    a circle. Each node renders the molecule inside a colored ring (ring color = target
    value), and edge width scales with similarity -- so an activity cliff (near-identical
    structures, very different target) reads at a glance.

    Args:
        query_id: Id of the center compound; must appear in `neighbors_df[id_col]`.
        neighbors_df: One row per compound, with id, smiles, similarity, and target
            columns. Must include the query's own row -- e.g. from
            `neighbors(..., include_self=True)`, where its similarity is 1.0.
        target_col: Column holding the target value; drives ring color (NaN -> gray).
        smiles_col (str): Column holding SMILES strings. Defaults to "smiles".
        id_col (str): Column holding compound ids. Defaults to "neighbor_id".
        similarity_col (str): Column holding similarity to the query, higher = closer.
            Defaults to "similarity".
        n_neighbors (int): How many closest neighbors to draw. Defaults to 5.
        cmap_name (str): Matplotlib colormap for the target ring/colorbar. Defaults
            to "viridis".
        title (str, optional): Plot title. A sensible default is built when None.

    Returns:
        matplotlib.figure.Figure: The graph figure. The caller shows or saves it:
            `fig.show()`, or `fig.savefig(path, dpi=150, bbox_inches="tight")`.
    """
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.cm import ScalarMappable

    df = neighbors_df.dropna(subset=[smiles_col]).copy()
    query_match = df[df[id_col] == query_id]
    if query_match.empty:
        raise ValueError(
            f"query id {query_id!r} not found in neighbors_df[{id_col!r}] -- include the "
            f"query's own row, e.g. neighbors(..., include_self=True)."
        )
    query_row = query_match.iloc[0]
    neigh = df[df[id_col] != query_id].nlargest(n_neighbors, similarity_col).reset_index(drop=True)
    ring_rows = pd.concat([query_row.to_frame().T, neigh], ignore_index=True)

    # Ring color scale over the query + drawn neighbors.
    vals = pd.to_numeric(ring_rows[target_col], errors="coerce")
    finite = vals.dropna()
    vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps[cmap_name]

    def ring_color(v):
        return "gray" if pd.isna(v) else cmap(norm(v))

    # Circular molecule image (white background masked to a disc).
    def circle_img(smiles, px=340):
        img = img_from_smiles(smiles, width=px, height=px, background="rgba(255,255,255,1)")
        if img is None:
            return np.zeros((px, px, 4), dtype=np.uint8)
        arr = np.array(img.convert("RGBA"))
        yy, xx = np.ogrid[:px, :px]
        r = px / 2
        mask = (xx - r) ** 2 + (yy - r) ** 2 <= (r - 2) ** 2
        arr[~mask, 3] = 0
        return arr

    fig, ax = plt.subplots(figsize=(11.5, 11))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_autoscale_on(False)  # keep imshow from rescaling the axes

    center = (0.0, 0.0)
    angles = np.linspace(90, 90 - 360, len(neigh), endpoint=False)
    radius = 1.15
    positions = {query_id: center}
    for row, ang in zip(neigh.itertuples(), angles):
        positions[getattr(row, id_col)] = (radius * np.cos(np.radians(ang)), radius * np.sin(np.radians(ang)))

    # Edge width scaled across the observed similarity range.
    sims = neigh[similarity_col]
    smin, smax = (float(sims.min()), float(sims.max())) if len(sims) else (0.0, 1.0)

    def edge_lw(sim):
        frac = (sim - smin) / (smax - smin) if smax > smin else 1.0
        return 1.5 + 11 * frac

    for row in neigh.itertuples():
        x, y = positions[getattr(row, id_col)]
        sim = getattr(row, similarity_col)
        ax.plot([center[0], x], [center[1], y], color="#888", lw=edge_lw(sim), zorder=1, alpha=0.7)
        ax.text(
            x * 0.5,
            y * 0.5,
            f"{sim:.2f}",
            fontsize=11,
            color="#333",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", fc="white", ec="#888"),
            zorder=2,
        )

    node_r, gap = 0.40, 0.03
    for nid, (x, y) in positions.items():
        row = ring_rows[ring_rows[id_col] == nid].iloc[0]
        tval = pd.to_numeric(row[target_col], errors="coerce")
        ax.add_patch(Circle((x, y), node_r, facecolor="white", edgecolor=ring_color(tval), lw=6, zorder=3))
        ax.imshow(circle_img(row[smiles_col]), extent=(x - node_r, x + node_r, y - node_r, y + node_r), zorder=4)
        tstr = "n/a" if pd.isna(tval) else f"{tval:.2f}"
        lbl = "QUERY" if nid == query_id else "neighbor"
        va, off = ("bottom", node_r + gap) if y > 0.5 else ("top", -node_r - gap)
        ax.text(x, y + off, f"{nid}\n{lbl}  {target_col}={tstr}", fontsize=10.5, ha="center", va=va, zorder=5)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
    cbar.set_label(target_col, fontsize=12)

    if title is None:
        title = f"{query_id} + {len(neigh)} closest neighbors\n" f"(ring = {target_col}, edge width = {similarity_col})"
    ax.set_title(title, fontsize=14, pad=16)
    fig.tight_layout()
    return fig
