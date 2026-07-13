/* ML Pipelines page renderer.
 *
 * A Dash clientside callback (namespace "ml_pipelines", function "render") calls
 * mlpRender(data) whenever the server-side pipeline data changes. The data is the
 * group tree from CachedMeta.pipelines(); each pipeline is an artifact-only lineage
 * node-link {nodes: [{id, type}], links: [{source, target}]} already collapsed and
 * threaded (ds -> fs -> model -> endpoint) by the server's linearize(). The renderer
 * just lays it out and draws it -- no graph semantics live here.
 *
 * We render straight into #ml-pipelines-root via the DOM (rather than returning Dash
 * components) so the interactive card -> graph flow lives entirely in the browser.
 */

(function () {
  const CHEVRON ='<svg viewBox="0 0 20 20" fill="none" width="18" height="18"><path d="M6 8l4 4 4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const KIND_COLOR = {
    feature_sets: "var(--mlp-t-fs)",
    class: "var(--mlp-t-model)",
    reg: "var(--mlp-accent)",
    other: "var(--mlp-text-faint)",
  };

  function kindOf(group) {
    if (group.endsWith("feature_sets")) return "feature_sets";
    if (group.includes("_class")) return "class";
    if (group.includes("_reg")) return "reg";
    return "other";
  }

  // Merge all pipeline graphs of an assay into one node-link graph (dedup nodes by id)
  function mergeGraphs(assay) {
    const nodes = new Map(); const links = [];
    Object.values(assay).forEach((g) => {
      (g.nodes || []).forEach((n) => { if (!nodes.has(n.id)) nodes.set(n.id, n); });
      (g.links || []).forEach((l) => links.push(l));
    });
    return { nodes: [...nodes.values()], links };
  }

  // Artifact-type counting. Types shown, in pipeline order, with display labels.
  const COUNT_TYPES = ["public", "ds", "fs", "model", "endpoint"];
  const TYPE_LABEL = { public: "PublicData", ds: "DataSource", fs: "FeatureSet", model: "Model", endpoint: "Endpoint" };

  // Dashboard page a node type drills into: /<route>?name=<artifact name> (new tab).
  // No route for `public` -- external public data has no dashboard page, so it isn't linked.
  const ARTIFACT_ROUTE = { ds: "data_sources", fs: "feature_sets", model: "models", endpoint: "endpoints" };

  // Unique artifact counts by type across one or more pipeline graphs
  function typeCounts(graphs) {
    const sets = { public: new Set(), ds: new Set(), fs: new Set(), model: new Set(), endpoint: new Set() };
    graphs.forEach((g) => (g.nodes || []).forEach((n) => {
      if (sets[n.type]) sets[n.type].add(n.id);
    }));
    const out = {};
    COUNT_TYPES.forEach((t) => { out[t] = sets[t].size; });
    return out;
  }
  // Every pipeline graph at or below a group (its own + all descendants')
  function allGraphs(group) {
    let gs = Object.values(group.pipelines || {});
    (group.subgroups || []).forEach((sg) => { gs = gs.concat(allGraphs(sg)); });
    return gs;
  }
  function groupCounts(group) { return typeCounts(allGraphs(group)); }
  const isLeaf = (group) => !group.subgroups || group.subgroups.length === 0;

  /* The single rule for what a container renders as cards vs. nested sections:
     every leaf child is a card; every non-leaf child is a section; the container's
     own non-empty pipelines are one more card. All cards at a level share `path`.
     Both the grid (renderContents) and the preview (collectCards) go through here so
     the two can never disagree on what a "card" is. */
  function splitLevel(groups, path, ownPipelines, ownName) {
    const cards = [], sections = [];
    (groups || []).forEach((g) => (isLeaf(g)
      ? cards.push({ name: g.name, pipelines: g.pipelines, path })
      : sections.push(g)));
    if (ownPipelines && Object.keys(ownPipelines).length) {
      cards.push({ name: ownName, pipelines: ownPipelines, path });
    }
    return { cards, sections };
  }

  // Flatten the whole tree to every card the grid would render, in the same order and
  // with the same breadcrumbs. Mirrors renderContents' recursion (sections recurse with
  // their name appended to the path and their own pipelines as this level's ownPipelines).
  function collectCards(groups, path, ownPipelines, ownName, out) {
    const { cards, sections } = splitLevel(groups, path, ownPipelines, ownName);
    cards.forEach((c) => out.push(c));
    sections.forEach((g) => collectCards(g.subgroups, path.concat(g.name), g.pipelines, g.name, out));
    return out;
  }

  // Pipeline complexity: product of (count + 1) across all 5 lanes. The +1 keeps an
  // empty lane from zeroing the product (most pipelines don't span all 5), and it
  // rewards both breadth (using more lanes) and balance (2,2,2,2,2 beats 1,1,1,4,3).
  function complexity(counts) {
    return COUNT_TYPES.reduce((p, t) => p * (counts[t] + 1), 1);
  }

  // Short labels for the compact card badges (avoid wrapping)
  const SHORT_LABEL = { public: "Pub", ds: "DS", fs: "FS", model: "Model", endpoint: "End" };

  // Long labels pluralize with "s" (except PublicData, a mass noun). Short labels are
  // fixed abbreviations except "Model" -> "Models".
  const longLabel = (t, n) => TYPE_LABEL[t] + (t !== "public" && n !== 1 ? "s" : "");
  const shortLabel = (t, n) => SHORT_LABEL[t] + (t === "model" && n !== 1 ? "s" : "");

  // Count pills (skips any type with a zero count), number colored by type.
  // short=true uses the compact abbreviations so the card badges fit on one row.
  function countPills(counts, short) {
    return COUNT_TYPES.filter((t) => counts[t] > 0).map((t) => {
      const label = short ? shortLabel(t, counts[t]) : longLabel(t, counts[t]);
      return `<span class="mlp-stat-pill"><b style="color:var(--mlp-t-${t})">${counts[t]}</b> ${label}</span>`;
    }).join("");
  }

  /* Shape the server's artifact-only lineage node-link for the layout engine:
     {nodes: [{id, type, name}], edges: [[a, b]]}. The collapse + type-ladder threading
     already happened in the server's linearize(); here we only derive display names and
     turn links into edge pairs (dropping any dangling/self edges defensively). */
  function buildFromGraph(graph) {
    const artifacts = new Map();
    (graph.nodes || []).forEach((n) => {
      const i = n.id.indexOf(":");
      artifacts.set(n.id, { id: n.id, type: n.type, name: i >= 0 ? n.id.slice(i + 1) : n.id });
    });
    const seen = new Set(); const edges = [];
    (graph.links || []).forEach(({ source, target }) => {
      const key = source + ">" + target;
      if (source !== target && artifacts.has(source) && artifacts.has(target) && !seen.has(key)) {
        seen.add(key); edges.push([source, target]);
      }
    });
    return { nodes: [...artifacts.values()], edges };
  }

  // Column band per artifact type: each column holds exactly one type, bands ordered
  // ds -> fs -> model -> endpoint. A derived fs (fs -> fs) gets its own extra fs column.
  const TYPE_ORDER = { ds: 0, public: 0, fs: 1, model: 2, endpoint: 3 };
  const typeBand = (t) => (TYPE_ORDER[t] != null ? TYPE_ORDER[t] : 2);

  /* Type-aware layered DAG layout. Columns are (type-band, same-type depth): every
     column is a single type; fs -> fs adds a second fs column. Barycenter ordering
     within each column minimizes edge crossings. Returns layout(id -> {rank, order})
     and per-column counts. */
  function layoutDAG(nodes, edges) {
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const succ = new Map(), pred = new Map();
    nodes.forEach((n) => { succ.set(n.id, []); pred.set(n.id, []); });
    edges.forEach(([a, b]) => { if (byId.has(a) && byId.has(b)) { succ.get(a).push(b); pred.get(b).push(a); } });

    // Topological order (Kahn) to compute same-type depth
    const indeg = new Map(nodes.map((n) => [n.id, pred.get(n.id).length]));
    const topo = nodes.filter((n) => indeg.get(n.id) === 0).map((n) => n.id);
    for (let h = 0; h < topo.length; h++) {
      succ.get(topo[h]).forEach((v) => {
        indeg.set(v, indeg.get(v) - 1);
        if (indeg.get(v) === 0) topo.push(v);
      });
    }
    // same-type depth: longest chain of same-band predecessors (splits fs -> fs)
    const band = (id) => typeBand(byId.get(id).type);
    const sdepth = new Map(nodes.map((n) => [n.id, 0]));
    topo.forEach((u) => pred.get(u).forEach((p) => {
      if (band(p) === band(u) && sdepth.get(u) < sdepth.get(p) + 1) sdepth.set(u, sdepth.get(p) + 1);
    }));
    // Column = sorted unique (band, sdepth) key -> sequential index
    const colKey = (id) => band(id) * 1000 + sdepth.get(id);
    const keys = [...new Set(nodes.map((n) => colKey(n.id)))].sort((a, b) => a - b);
    const realCol = new Map(nodes.map((n) => [n.id, keys.indexOf(colKey(n.id))]));

    // Insert dummy nodes for edges spanning >1 column so long edges route through
    // reserved slots (Sugiyama) — this is what lets the ordering remove crossings.
    const layers = Array.from({ length: keys.length }, () => []);
    nodes.forEach((n) => layers[realCol.get(n.id)].push(n.id));
    const up = new Map(), down = new Map();
    const link = (u, v) => { (down.get(u) || down.set(u, []).get(u)).push(v); (up.get(v) || up.set(v, []).get(v)).push(u); };
    nodes.forEach((n) => { up.set(n.id, []); down.set(n.id, []); });
    const routes = [];
    let dummySeq = 0;
    edges.forEach(([a, b]) => {
      const ca = realCol.get(a), cb = realCol.get(b);
      if (ca == null || cb == null) return;
      const chain = [];
      for (let c = ca + 1; c < cb; c++) {
        const dk = ">d" + dummySeq++;
        up.set(dk, []); down.set(dk, []); layers[c].push(dk); chain.push(dk);
      }
      const seq = [a, ...chain, b];
      for (let i = 0; i < seq.length - 1; i++) link(seq[i], seq[i + 1]);
      routes.push({ a, b, chain });
    });

    // Column ordering (crossing reduction). Three cheap passes:
    //  1. Baseline: order each column by the median position of its inputs.
    //  2. Compound: middle columns (with both inputs and outputs) re-sort by a compound
    //     key (input position, then output position). This is what orders the Models
    //     lane so models fed by the same feature set AND feeding the same endpoint sit
    //     together — cleaning up both the fan-out and the fan-in.
    //  3. Terminal: re-order the last column by its now-updated inputs.
    const orderMap = (layer) => { const m = new Map(); layer.forEach((id, i) => m.set(id, i)); return m; };
    const median = (xs) => {
      if (!xs.length) return null;
      const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
      return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    };
    const medAt = (id, neigh, ord) => median(neigh.get(id).map((x) => ord.get(x)).filter((v) => v != null));
    for (let r = 1; r < layers.length; r++) {
      const ord = orderMap(layers[r - 1]);
      layers[r] = layers[r].map((id, i) => ({ id, i, k: medAt(id, up, ord) }))
        .sort((a, b) => (a.k ?? a.i) - (b.k ?? b.i) || a.i - b.i).map((o) => o.id);
    }
    for (let r = 1; r < layers.length - 1; r++) {
      const uo = orderMap(layers[r - 1]), dvo = orderMap(layers[r + 1]);
      layers[r] = layers[r].map((id, i) => ({ id, i, ik: medAt(id, up, uo), ok: medAt(id, down, dvo) }))
        .sort((a, b) => (a.ik ?? a.i) - (b.ik ?? b.i) || (a.ok ?? a.i) - (b.ok ?? b.i) || a.i - b.i)
        .map((o) => o.id);
    }
    if (layers.length > 1) {
      const r = layers.length - 1, ord = orderMap(layers[r - 1]);
      layers[r] = layers[r].map((id, i) => ({ id, i, k: medAt(id, up, ord) }))
        .sort((a, b) => (a.k ?? a.i) - (b.k ?? b.i) || a.i - b.i).map((o) => o.id);
    }
    // Vertical coordinate assignment (in gap units). Anchor the tallest column with
    // even spacing, then place every other column by the median of its already-placed
    // neighbors: columns to the RIGHT align to their inputs (so endpoints sit at the
    // median of the models feeding them), columns to the LEFT align to their outputs
    // (so feature sets sit at the median of the models they feed). This gives the
    // tree-like fan-out/fan-in without sparse columns clustering in the middle.
    const yc = new Map();
    const medY = (id, neigh) => {
      const ys = neigh.get(id).map((x) => yc.get(x)).filter((v) => v != null).sort((a, b) => a - b);
      if (!ys.length) return null;
      const m = ys.length >> 1;
      return ys.length % 2 ? ys[m] : (ys[m - 1] + ys[m]) / 2;
    };
    // Anchor = tallest column, evenly spaced in its established order
    let anchor = 0;
    layers.forEach((l, i) => { if (l.length > layers[anchor].length) anchor = i; });
    layers[anchor].forEach((id, i) => yc.set(id, i));
    // Place a column near its neighbors' medians, preserving order + min gap, then
    // shift the whole column so its centroid sits on the neighbors' centroid.
    const place = (col, neigh) => {
      const layer = layers[col];
      const desired = layer.map((id) => medY(id, neigh));
      const y = []; let prev = -Infinity;
      for (let i = 0; i < layer.length; i++) {
        let d = desired[i] == null ? (i > 0 ? y[i - 1] + 1 : 0) : desired[i];
        d = Math.max(d, prev + 1); y[i] = d; prev = d;
      }
      const known = desired.filter((v) => v != null);
      if (known.length) {
        const shift = known.reduce((s, v) => s + v, 0) / known.length - y.reduce((s, v) => s + v, 0) / y.length;
        for (let i = 0; i < y.length; i++) y[i] += shift;
      }
      layer.forEach((id, i) => yc.set(id, y[i]));
    };
    for (let c = anchor + 1; c < layers.length; c++) place(c, up);   // right: by inputs
    for (let c = anchor - 1; c >= 0; c--) place(c, down);            // left: by outputs
    const minY = Math.min(...yc.values());
    yc.forEach((v, k) => yc.set(k, v - minY));
    const maxY = Math.max(0, ...yc.values());

    const layout = new Map();
    layers.forEach((layer, r) => layer.forEach((id) => { if (byId.has(id)) layout.set(id, { rank: r, y: yc.get(id) }); }));
    const edgeRoutes = routes.map(({ a, b, chain }) => ({ a, b, waypoints: chain.map((dk) => ({ rank: realCol.get(a) + 1 + chain.indexOf(dk), y: yc.get(dk) })) }));
    return { layout, edgeRoutes, cols: layers.length, maxY };
  }

  const SVGNS = "http://www.w3.org/2000/svg";

  // Node box width sized to its name (monospace char width estimate), clamped
  function nodeWidthFor(name) {
    // +30 = 11px padding each side + 4px endcap border each side (a bit of slack so
    // names don't ellipsize); *7.0 approximates the 11px mono char advance.
    return Math.max(140, Math.min(320, Math.ceil((name || "").length * 7.0) + 30));
  }

  function renderDAG(graph, availWidth) {
    const { nodes, edges } = buildFromGraph(graph);
    const { layout, edgeRoutes, cols, maxY } = layoutDAG(nodes, edges);
    const nodeH = 36, rowStep = 50, pad = 16, colGap = 46;

    // Per-node width (dynamic); the widest box drives column spacing
    const nodeW = new Map(nodes.map((n) => [n.id, nodeWidthFor(n.name)]));
    const maxNodeW = Math.max(130, ...nodeW.values());
    const minColStep = maxNodeW + colGap;
    const colStep = cols > 1 ? Math.max(minColStep, (availWidth - maxNodeW - pad * 2) / (cols - 1)) : 0;
    const width = maxNodeW + colStep * (cols - 1) + pad * 2;
    const height = pad * 2 + nodeH + maxY * rowStep;

    const colLeft = (rank) => pad + rank * colStep;
    const wpX = (rank) => colLeft(rank) + maxNodeW / 2;
    const yTop = (u) => pad + u * rowStep;
    const yMid = (u) => pad + u * rowStep + nodeH / 2;
    const pos = {};
    nodes.forEach((n) => {
      const { rank, y } = layout.get(n.id);
      pos[n.id] = { x: colLeft(rank), y: yTop(y), w: nodeW.get(n.id) };
    });

    const dag = document.createElement("div");
    dag.className = "mlp-dag";
    dag.style.width = width + "px";
    dag.style.height = height + "px";
    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("class", "mlp-edges");
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);

    // Draw each edge as a smooth spline through its (dummy) waypoints
    const routeByPair = new Map(edgeRoutes.map((r) => [r.a + ">" + r.b, r.waypoints]));
    const nbrs = new Map(nodes.map((n) => [n.id, new Set()]));
    const edgeEls = [];
    edges.forEach(([a, b]) => {
      const pa = pos[a], pb = pos[b]; if (!pa || !pb) return;
      nbrs.get(a).add(b); nbrs.get(b).add(a);
      const pts = [{ x: pa.x + pa.w, y: pa.y + nodeH / 2 }];
      (routeByPair.get(a + ">" + b) || []).forEach((wp) => pts.push({ x: wpX(wp.rank), y: yMid(wp.y) }));
      pts.push({ x: pb.x, y: pb.y + nodeH / 2 });
      let d = `M ${pts[0].x} ${pts[0].y}`;
      for (let i = 1; i < pts.length; i++) {
        const p0 = pts[i - 1], p1 = pts[i], mx = (p0.x + p1.x) / 2;
        d += ` C ${mx} ${p0.y}, ${mx} ${p1.y}, ${p1.x} ${p1.y}`;
      }
      const p = document.createElementNS(SVGNS, "path");
      p.setAttribute("d", d);
      p.setAttribute("fill", "none");
      p.setAttribute("stroke", "var(--mlp-edge)");
      p.setAttribute("stroke-width", "1.4");
      svg.appendChild(p);
      edgeEls.push({ a, b, el: p });
    });
    dag.appendChild(svg);

    const nodeEls = new Map();
    nodes.forEach((n) => {
      const el = document.createElement("div");
      el.className = "mlp-node n-" + n.type;
      el.style.left = pos[n.id].x + "px";
      el.style.top = pos[n.id].y + "px";
      el.style.width = pos[n.id].w + "px";
      el.style.setProperty("--nc", `var(--mlp-t-${n.type})`);
      el.addEventListener("mouseenter", () => setHover(n.id, true));
      el.addEventListener("mouseleave", () => setHover(n.id, false));
      // Drill-down: click opens the artifact's dashboard page in a new tab.
      const route = ARTIFACT_ROUTE[n.type];
      if (route) {
        el.classList.add("mlp-node-link");
        el.innerHTML = `<div class="n-name" title="${n.id} — open in new tab">${n.name}</div>`;
        el.addEventListener("click", () =>
          window.open(`/${route}?name=${encodeURIComponent(n.name)}`, "_blank", "noopener"));
      } else {
        el.innerHTML = `<div class="n-name" title="${n.id}">${n.name}</div>`;
      }
      nodeEls.set(n.id, el);
      dag.appendChild(el);
    });

    // Hover: outline the node in its type color, dim the rest, light its edges + neighbors
    function setHover(id, on) {
      dag.classList.toggle("mlp-hovering", on);
      nodeEls.get(id).classList.toggle("mlp-hl", on);
      (nbrs.get(id) || new Set()).forEach((o) => nodeEls.get(o) && nodeEls.get(o).classList.toggle("mlp-nb", on));
      edgeEls.forEach((e) => { if (e.a === id || e.b === id) e.el.classList.toggle("mlp-edge-hl", on); });
    }
    return dag;
  }

  // A faithful scaled-down version of the drill-down layout: same routed layout
  // (dummies on) and same waypoint edge splines, just mapped into the thumbnail box.
  function miniThumb(assay) {
    const { nodes, edges } = buildFromGraph(mergeGraphs(assay));
    const { layout, edgeRoutes, cols, maxY } = layoutDAG(nodes, edges);
    const W = 268, H = 76, pad = 8;
    const xAt = (rank) => (cols > 1 ? pad + (rank / (cols - 1)) * (W - pad * 2) : W / 2);
    const yAt = (u) => (maxY > 0 ? pad + (u / maxY) * (H - pad * 2) : H / 2);
    const pos = {};
    nodes.forEach((n) => { const { rank, y } = layout.get(n.id); pos[n.id] = { x: xAt(rank), y: yAt(y) }; });

    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "mlp-thumb");
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    const routeByPair = new Map(edgeRoutes.map((r) => [r.a + ">" + r.b, r.waypoints]));
    edges.forEach(([a, b]) => {
      const pa = pos[a], pb = pos[b]; if (!pa || !pb) return;
      const pts = [pa];
      (routeByPair.get(a + ">" + b) || []).forEach((wp) => pts.push({ x: xAt(wp.rank), y: yAt(wp.y) }));
      pts.push(pb);
      let d = `M ${pts[0].x} ${pts[0].y}`;
      for (let i = 1; i < pts.length; i++) {
        const p0 = pts[i - 1], p1 = pts[i], mx = (p0.x + p1.x) / 2;
        d += ` C ${mx} ${p0.y}, ${mx} ${p1.y}, ${p1.x} ${p1.y}`;
      }
      const p = document.createElementNS(SVGNS, "path");
      p.setAttribute("d", d);
      p.setAttribute("fill", "none");
      p.setAttribute("stroke", "var(--mlp-edge)");
      p.setAttribute("stroke-width", "1");
      svg.appendChild(p);
    });
    const nw = 15, nh = 7;  // pill: fully rounded (rx = nh/2); nh ~ old dot diameter to avoid stacking overlap
    nodes.forEach((n) => {
      const r = document.createElementNS(SVGNS, "rect");
      r.setAttribute("x", pos[n.id].x - nw / 2);
      r.setAttribute("y", pos[n.id].y - nh / 2);
      r.setAttribute("width", nw);
      r.setAttribute("height", nh);
      r.setAttribute("rx", nh / 2);
      r.setAttribute("ry", nh / 2);
      r.setAttribute("fill", `var(--mlp-t-${n.type})`);
      svg.appendChild(r);
    });
    return svg;
  }

  function legendHTML() {
    return `<div class="mlp-legend">
      <span class="mlp-type-chip t-public"><span class="mlp-dot bg-public"></span>PublicData</span>
      <span class="mlp-type-chip t-ds"><span class="mlp-dot bg-ds"></span>DataSource</span>
      <span class="mlp-type-chip t-fs"><span class="mlp-dot bg-fs"></span>FeatureSet</span>
      <span class="mlp-type-chip t-model"><span class="mlp-dot bg-model"></span>Model</span>
      <span class="mlp-type-chip t-endpoint"><span class="mlp-dot bg-endpoint"></span>Endpoint</span>
    </div>`;
  }

  function buildGrid(root, data) {
    const grid = document.createElement("div");
    grid.className = "mlp-grid";
    grid.innerHTML = `<div class="mlp-toolbar">
        <button class="mlp-collapse-all">Collapse all</button>
        ${legendHTML()}
      </div>`;

    // Top level renders like any group's contents: leaf groups become cards, deeper
    // groups become nested collapsible sections. `path` is the ancestor breadcrumb.
    renderContents(root, data, grid, data, [], 0, null, null);

    grid.querySelector(".mlp-collapse-all").onclick = (e) => {
      const sections = [...grid.querySelectorAll(".mlp-cat")];
      const anyOpen = sections.some((s) => !s.classList.contains("collapsed"));
      sections.forEach((s) => s.classList.toggle("collapsed", anyOpen));
      e.target.textContent = anyOpen ? "Expand all" : "Collapse all";
    };
    return grid;
  }

  /* Render a group's contents into `container`: deeper groups as nested sections,
     then leaf groups (and any own pipelines) as cards in one grid. Sections render
     before cards (folders-first) so a loose top-level group like "Misc" sinks to the
     bottom. Groups arrive name-sorted from the serializer. Depth-agnostic. */
  function renderContents(root, data, container, groups, path, depth, ownPipelines, ownName) {
    const { cards, sections } = splitLevel(groups, path, ownPipelines, ownName);
    sections.forEach((g) => container.appendChild(renderSection(root, data, g, path, depth)));
    if (cards.length) {
      const cardGrid = document.createElement("div");
      cardGrid.className = "mlp-card-grid";
      cards.forEach((c) => cardGrid.appendChild(makeCard(root, data, c.name, c.pipelines, c.path)));
      container.appendChild(cardGrid);
    }
  }

  function renderSection(root, data, group, parentPath, depth) {
    const path = parentPath.concat(group.name);
    const section = document.createElement("div");
    section.className = "mlp-cat";
    section.dataset.depth = depth;
    const head = document.createElement("div");
    head.className = "mlp-cat-head";
    head.innerHTML = `<span class="mlp-caret">${CHEVRON}</span>
      <h3>${group.name}</h3>
      <span class="mlp-cs-count">${countPills(groupCounts(group))}</span>`;
    const body = document.createElement("div");
    body.className = "mlp-cat-body";
    const inner = document.createElement("div");
    inner.className = "mlp-cat-body-inner";
    body.appendChild(inner);
    section.append(head, body);
    renderContents(root, data, inner, group.subgroups, path, depth + 1, group.pipelines, group.name);
    head.onclick = () => section.classList.toggle("collapsed");
    return section;
  }

  // Build the card DOM (title + breadcrumb + thumbnail + stat pills). The caller
  // attaches the onclick (grid cards open the detail view; preview cards navigate).
  function cardEl(name, pipelines, path, counts) {
    const card = document.createElement("div");
    card.className = "mlp-card";
    const sub = path.length ? path.join(" / ") : "";
    card.innerHTML = `
      <div class="mlp-card-top">
        <div>
          <div class="mlp-card-title">${name}</div>
          ${sub ? `<div class="mlp-card-sub">${sub}</div>` : ""}
        </div>
        <span class="mlp-card-open">Details →</span>
      </div>`;
    card.appendChild(miniThumb(pipelines));
    const stats = document.createElement("div");
    stats.className = "mlp-card-stats";
    stats.innerHTML = countPills(counts, true);
    card.appendChild(stats);
    return card;
  }

  function makeCard(root, data, name, pipelines, path) {
    const card = cardEl(name, pipelines, path, typeCounts(Object.values(pipelines)));
    card.onclick = () => openDetail(root, data, name, pipelines, path);
    return card;
  }

  function openDetail(root, data, name, pipelines, path) {
    const counts = typeCounts(Object.values(pipelines));
    const merged = mergeGraphs(pipelines);
    const groupTabs = Object.keys(pipelines).map((g) => {
      const k = kindOf(g);
      return `<span class="mlp-group-tab"><span class="mlp-gt-kind" style="background:${KIND_COLOR[k]}"></span>${g}</span>`;
    }).join("");
    const metrics = COUNT_TYPES.filter((t) => counts[t] > 0).map((t) => {
      const label = longLabel(t, counts[t]);
      return `<div class="mlp-metric"><div class="mv" style="color:var(--mlp-t-${t})">${counts[t]}</div><div class="ml">${label}</div></div>`;
    }).join("");
    const eyebrow = path.length ? path.join(" / ") : "";

    const detail = document.createElement("div");
    detail.className = "mlp-detail";
    detail.innerHTML = `
      <div class="mlp-detail-bar">
        <button class="mlp-back">
          <span class="mlp-arrow"><svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 3.5L5.5 8l4.5 4.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
          Pipelines
        </button>
        <div class="mlp-detail-title">
          ${eyebrow ? `<span class="mlp-cat-eyebrow">${eyebrow}</span>` : ""}
          <h2>${name}</h2>
        </div>
        <div class="mlp-detail-metrics">${metrics}</div>
      </div>
      <div class="mlp-canvas">
        <div class="mlp-canvas-top">
          <div class="mlp-group-tabs">${groupTabs}</div>
          <div style="margin-left:auto">${legendHTML()}</div>
        </div>
        <div class="mlp-detail-dag"></div>
      </div>`;
    // Both the "< Pipelines" button and the browser back button route through
    // history.back() -> popstate -> showGrid, so muscle memory works either way.
    detail.querySelector(".mlp-back").onclick = () => history.back();

    root.replaceChildren(detail);
    window.scrollTo({ top: 0, behavior: "smooth" });

    // Track the open pipeline so a browser resize can re-fill the canvas
    activeDetail = { detail, merged };
    layoutDetailDag();
    detailResizeObserver.observe(detail.querySelector(".mlp-canvas"));  // re-fit on any resize

    // Push a history entry (URL unchanged, so Dash's router is untouched) so the
    // browser back button returns to the grid instead of leaving the page.
    history.pushState({ mlpDetail: name }, "", window.location.href);
  }

  // (Re)lay the active detail DAG so it fits the canvas like the card thumbnails do:
  // render at natural size, then uniformly scale the whole graph to fit the available
  // box (width AND height). Capped at 1x so wide screens keep normal-size nodes (the
  // column layout already spreads to fill width) while narrow/short ones shrink to fit
  // -- no clipping, no scrolling.
  function layoutDetailDag() {
    if (!activeDetail) return;
    const { detail, merged } = activeDetail;
    const canvas = detail.querySelector(".mlp-canvas");
    const host = detail.querySelector(".mlp-detail-dag");
    if (!canvas || !host) return;
    const availW = canvas.clientWidth - 36;
    const dag = renderDAG(merged, availW);
    const natW = parseFloat(dag.style.width), natH = parseFloat(dag.style.height);
    if (!natW || !natH) { host.replaceChildren(dag); return; }
    // Fit within the box (contain); never upscale past 1x.
    const scale = Math.min(1, availW / natW, host.clientHeight / natH);
    dag.style.transformOrigin = "top left";
    dag.style.transform = `scale(${scale})`;
    // Wrapper reserves the *scaled* footprint so flex centering positions it correctly
    // (a CSS transform doesn't change the element's layout box).
    const scaler = document.createElement("div");
    scaler.className = "mlp-dag-scaler";
    scaler.style.width = natW * scale + "px";
    scaler.style.height = natH * scale + "px";
    scaler.appendChild(dag);
    host.replaceChildren(scaler);
  }

  function showGrid(root, data) {
    activeDetail = null;
    detailResizeObserver.disconnect();
    root.replaceChildren(buildGrid(root, data));
  }

  // Responsive: re-fit the detail DAG whenever its canvas changes size -- window resize,
  // sidebar toggle, or any layout shift. A ResizeObserver on the element catches size
  // changes a window 'resize' listener misses; the grid view self-sizes via CSS.
  let activeDetail = null, resizeTimer = null, currentRoot = null, currentData = null;
  const detailResizeObserver = new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (activeDetail) layoutDetailDag(); }, 120);
  });

  // Browser back (or the "< Pipelines" button): if a detail is open, pop back to
  // the grid. When no detail is open, let the back navigation leave the page.
  window.addEventListener("popstate", () => {
    if (activeDetail && currentRoot && currentData) showGrid(currentRoot, currentData);
  });

  /* Render entrypoint: called by the Dash clientside callback. Skips a re-render when
     the data signature is unchanged so the 60s refresh doesn't disturb a graph the
     user is viewing. */
  function mlpRender(data) {
    const root = document.getElementById("ml-pipelines-root");
    if (!root || !data) return "";
    currentRoot = root;
    currentData = data;
    const sig = JSON.stringify(data);
    if (root.dataset.sig === sig) return root.dataset.sig;
    root.dataset.sig = sig;
    showGrid(root, data);
    return String(sig.length);
  }

  /* Main-page preview: render the 4 most complex pipeline cards into #mlp-preview-root.
     Cards look identical to the grid's, but clicking one navigates to the full
     ML Pipelines page instead of drilling into a detail view. */
  function mlpRenderPreview(data) {
    const host = document.getElementById("mlp-preview-root");
    if (!host || !data) return "";
    const sig = JSON.stringify(data);
    if (host.dataset.sig === sig) return host.dataset.sig;
    host.dataset.sig = sig;

    const cards = collectCards(data, [], null, null, []);
    cards.forEach((c) => { c.counts = typeCounts(Object.values(c.pipelines)); c.score = complexity(c.counts); });
    cards.sort((a, b) => b.score - a.score);

    const grid = document.createElement("div");
    grid.className = "mlp-card-grid mlp-preview-grid";
    cards.slice(0, 4).forEach((c) => {
      const card = cardEl(c.name, c.pipelines, c.path, c.counts);
      card.onclick = () => window.location.assign("/ml_pipelines");
      grid.appendChild(card);
    });
    host.replaceChildren(grid);
    return String(sig.length);
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    ml_pipelines: { render: mlpRender, renderPreview: mlpRenderPreview },
  });
})();
