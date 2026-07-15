/* Model Contests page renderer.
 *
 * A Dash clientside callback (namespace "contests", function "render") calls
 * ctRender(data) whenever the server-side contest data changes. The data is a list
 * of contests {group, rows} (from Reports() /contests/* plus the pipeline hierarchy
 * group): rows are the report's row dicts — champion first, then challengers ranked
 * best-first, with metric columns interleaved with Δ-vs-champion columns (positive =
 * challenger better). The renderer derives the contested flag and all coloring from
 * the rows.
 *
 * We render straight into #contests-root via the DOM so the card -> expand-in-place
 * interaction lives entirely in the browser (no server round-trips).
 */

(function () {
  const CHEVRON = '<svg viewBox="0 0 20 20" fill="none" width="18" height="18"><path d="M6 8l4 4 4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  // Columns that are contest metadata rather than metrics
  const META_COLS = new Set(["model", "role", "framework", "endpoint", "inference_run", "timestamp"]);

  // Deltas below this are run-to-run float noise (a champion vs its own frozen copy),
  // not a real difference: they display as 0 and never mark a contest "contested".
  const EPS = 1e-6;

  // Model framework: the report's `framework` column is authoritative (written by
  // contest_report(), multi-task already resolved). Unrecognized values map to "other".
  const FW_KEY = {
    "multi-task": "mt",
    chemprop: "chemprop",
    xgboost: "xgb",
    pytorch: "pytorch",
    transformer: "transformer",
    sklearn: "sklearn",
    meta: "meta",
  };
  const FRAMEWORK_LABEL = {
    mt: "multi-task", chemprop: "chemprop", xgb: "xgboost", pytorch: "pytorch",
    transformer: "transformer", sklearn: "sklearn", meta: "meta", other: "other",
  };
  const frameworkOf = (row) => FW_KEY[row.framework] || "other";

  const CROWN = '<svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M1.5 5.5l3.2 2.6L8 3.5l3.3 4.6 3.2-2.6-1.2 7H2.7l-1.2-7z"/></svg>';

  // Rank markers, in contest order: gold crown (champion), then silver, bronze,
  // and open circles for everyone after that. Shared by the ladder and the table.
  const MEDALS = ["silver", "bronze"];
  function rankMarker(role, challengerIdx) {
    if (role === "champion") return `<span class="ct-crown">${CROWN}</span>`;
    const medal = MEDALS[challengerIdx];
    return `<span class="ct-medal ${medal ? "ct-medal-" + medal : "ct-medal-open"}"></span>`;
  }

  // Reshape one contest ({group, rows}) into the card's working form
  function parseContest(contest) {
    const rows = contest.rows || [];
    const group = contest.group || [];
    const champion = rows.find((r) => r.role === "champion") || null;
    const challengers = rows.filter((r) => r.role === "challenger");
    const first = rows[0] || {};
    const primary = "rmse" in first ? "rmse" : "f1";
    const top = challengers[0];
    const topDelta = top && top["Δ" + primary] != null ? top["Δ" + primary] : null;
    return {
      rows,
      group,
      champion,
      challengers,
      endpoint: first.endpoint,
      run: first.inference_run,
      timestamp: first.timestamp,
      contested: topDelta != null && topDelta > EPS,
    };
  }

  // ---------- formatting ----------
  // Floats show 2 significant digits; integers (e.g. support counts) stay exact
  const fmt = (v) => {
    if (v == null || Number.isNaN(v)) return "—";
    const n = Number(v);
    return Number.isInteger(n) ? String(n) : String(parseFloat(n.toPrecision(2)));
  };
  // Metric values color by their Δ-vs-champion sign; deltas this small count as a tie
  const zeroish = (v) => v == null || Math.abs(v) < 0.005;
  const deltaClass = (row, metric, isChampion) => {
    const d = row["Δ" + metric];
    if (isChampion || d == null || zeroish(d)) return "";
    return d > 0 ? "ct-pos" : "ct-neg";
  };

  function timeAgo(iso) {
    const t = new Date(iso);
    if (isNaN(t)) return "";
    const mins = Math.max(0, Math.round((Date.now() - t.getTime()) / 60000));
    if (mins < 60) return mins + "m ago";
    if (mins < 60 * 24) return Math.round(mins / 60) + "h ago";
    return Math.round(mins / (60 * 24)) + "d ago";
  }

  // ---------- ladder (mini ranking) ----------
  const LADDER_MAX = 3; // challengers shown on the collapsed card

  // Column titles: "precision" shortens to "prec" so the header fits its column
  const metricLabel = (k) => k.replace("precision", "prec");

  // The contest's top 2 metrics (report column order, support excluded) shown as
  // numeric columns on each ladder row, colored by their Δ-vs-champion sign.
  function ladderMetrics(c) {
    return Object.keys(c.rows[0])
      .filter((k) => !META_COLS.has(k) && !k.startsWith("Δ") && k !== "support")
      .slice(0, 2);
  }

  function ladderRow(row, cls, marker, cols) {
    const fw = frameworkOf(row);
    const values = cols
      .map((k) => `<span class="ct-lval ${deltaClass(row, k, cls === "champ")}">${fmt(row[k])}</span>`)
      .join("");
    return `<div class="ct-lrow ${cls}">
      ${marker}
      <span class="ct-lname" title="${row.model}">${row.model}</span>
      <span class="ct-dot" style="background:var(--ct-f-${fw})" title="${FRAMEWORK_LABEL[fw]}"></span>
      ${values}
    </div>`;
  }

  function ladderHTML(c) {
    const cols = ladderMetrics(c);
    const rows = [];
    if (c.champion) rows.push(ladderRow(c.champion, "champ", rankMarker("champion", 0), cols));
    c.challengers.slice(0, LADDER_MAX).forEach((r, i) => {
      rows.push(ladderRow(r, "chall", rankMarker("challenger", i), cols));
    });
    const extra = c.challengers.length - LADDER_MAX;
    const more = extra > 0 ? `<div class="ct-more">+${extra} more challenger${extra > 1 ? "s" : ""}</div>` : "";
    const heads = cols.map((k) => `<span>${metricLabel(k)}</span>`).join("");
    return `<div class="ct-ladder" style="--ct-lcols:${cols.length}"><div class="ct-lhead">
        <span></span><span></span><span></span>${heads}
      </div>${rows.join("")}${more}</div>`;
  }

  // ---------- expanded table (full contest report) ----------
  function tableHTML(c) {
    const cols = Object.keys(c.rows[0]).filter((k) => !META_COLS.has(k) && !k.startsWith("Δ"));
    const head = `<tr><th></th><th class="ct-ta-l">model</th><th class="ct-ta-l">type</th>${cols
      .map((k) => `<th>${metricLabel(k)}</th>`)
      .join("")}</tr>`;
    let challengerIdx = 0;
    const body = c.rows
      .map((r) => {
        const cells = cols
          .map((k) => `<td class="${deltaClass(r, k, r.role === "champion")}">${fmt(r[k])}</td>`)
          .join("");
        const marker = rankMarker(r.role, r.role === "champion" ? 0 : challengerIdx++);
        const fw = frameworkOf(r);
        return `<tr class="${r.role === "champion" ? "ct-champ-row" : ""}">
          <td class="ct-rank">${marker}</td>
          <td class="ct-ta-l ct-model">${r.model}</td>
          <td class="ct-ta-l ct-type"><span class="ct-dot" style="background:var(--ct-f-${fw})"></span>${FRAMEWORK_LABEL[fw]}</td>${cells}</tr>`;
      })
      .join("");
    return `<div class="ct-table-wrap"><table class="ct-table">${head}${body}</table></div>`;
  }

  // ---------- cards ----------
  function makeCard(c) {
    const card = document.createElement("div");
    card.className = "ct-card" + (c.contested ? " contested" : "");
    card.innerHTML = `
      <div class="ct-card-top">
        <div>
          <div class="ct-card-title">${c.endpoint}</div>
          <div class="ct-card-sub">champion: ${c.champion ? c.champion.model : "—"}</div>
        </div>
        ${c.contested ? '<span class="ct-badge">contested</span>' : ""}
      </div>
      ${ladderHTML(c)}
      <div class="ct-detail">${tableHTML(c)}</div>
      <div class="ct-card-foot">
        <span class="ct-pill">${c.challengers.length} challenger${c.challengers.length !== 1 ? "s" : ""}</span>
        <span class="ct-pill">${c.run}</span>
        <span class="ct-foot-time" title="${c.timestamp}">${timeAgo(c.timestamp)}</span>
        <span class="ct-card-open">Expand →</span>
      </div>`;
    card.onclick = (e) => {
      // Clicks inside the expanded table shouldn't collapse (text selection, reading)
      if (card.classList.contains("expanded") && e.target.closest(".ct-detail")) return;
      const open = card.classList.toggle("expanded");
      card.querySelector(".ct-card-open").textContent = open ? "Collapse →" : "Expand →";
    };
    return card;
  }

  function cardGridEl(list) {
    const grid = document.createElement("div");
    grid.className = "ct-card-grid";
    list.forEach((c) => grid.appendChild(makeCard(c)));
    return grid;
  }

  // Collapsible group section (same interaction as the ML Pipelines page sections)
  function makeSection(title, list) {
    const contested = list.filter((c) => c.contested).length;
    const section = document.createElement("div");
    section.className = "ct-cat";
    const head = document.createElement("div");
    head.className = "ct-cat-head";
    head.innerHTML = `<span class="ct-caret">${CHEVRON}</span>
      <h3>${title}</h3>
      <span class="ct-pill">${list.length} contest${list.length !== 1 ? "s" : ""}</span>
      ${contested ? `<span class="ct-badge">${contested} contested</span>` : ""}`;
    const body = document.createElement("div");
    body.className = "ct-cat-body";
    const inner = document.createElement("div");
    inner.className = "ct-cat-body-inner";
    inner.appendChild(cardGridEl(list));
    body.appendChild(inner);
    section.append(head, body);
    head.onclick = () => section.classList.toggle("collapsed");
    return section;
  }

  // Framework legend chips for the frameworks actually present in the data
  function legendEl(contests) {
    const present = new Set();
    contests.forEach((c) => c.rows.forEach((r) => present.add(frameworkOf(r))));
    const order = ["chemprop", "mt", "xgb", "pytorch", "transformer", "sklearn", "meta", "other"].filter((fw) =>
      present.has(fw)
    );
    const legend = document.createElement("div");
    legend.className = "ct-legend";
    legend.innerHTML = order
      .map((fw) => `<span class="ct-legend-chip"><span class="ct-dot" style="background:var(--ct-f-${fw})"></span>${FRAMEWORK_LABEL[fw]}</span>`)
      .join("");
    return legend;
  }

  function buildGrid(root, data) {
    const contests = data.map(parseContest).filter((c) => c.rows.length);
    // Newest first within a group; contested ties break toward the top
    contests.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp) || b.contested - a.contested);

    if (!contests.length) {
      root.innerHTML = `<div class="ct-empty">No contest reports published yet.
        The promotion arbiter publishes results to <code>Reports()</code> at
        <code>/contests/&lt;endpoint&gt;</code> on every run.</div>`;
      return;
    }

    // Group by top-level pipeline hierarchy group (the ML Pipelines page's sections);
    // deeper path levels would give one section per card since each leaf group has one
    // contest endpoint. Ungrouped contests sink to the bottom as "Other"; a single
    // group renders flat.
    const groups = new Map();
    contests.forEach((c) => {
      const key = c.group[0] || "";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(c);
    });
    const container = document.createElement("div");
    container.className = "ct-grid";
    const toolbar = document.createElement("div");
    toolbar.className = "ct-toolbar";
    toolbar.appendChild(legendEl(contests));
    container.appendChild(toolbar);
    if (groups.size <= 1) {
      container.appendChild(cardGridEl(contests));
    } else {
      [...groups.keys()].sort((a, b) => (a === "") - (b === "") || a.localeCompare(b)).forEach((key) => {
        container.appendChild(makeSection(key || "Other", groups.get(key)));
      });
    }
    root.replaceChildren(container);
  }

  /* Render entrypoint: called by the Dash clientside callback. Skips a re-render when
     the data signature is unchanged so the periodic refresh doesn't collapse a card
     the user has expanded. */
  function ctRender(data) {
    const root = document.getElementById("contests-root");
    if (!root || !data) return "";
    const sig = JSON.stringify(data);
    if (root.dataset.sig === sig) return root.dataset.sig;
    root.dataset.sig = sig;
    buildGrid(root, data);
    return String(sig.length);
  }

  const PREVIEW_MAX = 3;

  // Fisher–Yates shuffle (used to fill the preview's non-contested slots at random)
  function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  /* Main-page preview: render up to 3 contests into #contests-preview-root — every
     contested contest first, then random others to fill. Cards look identical to the
     grid's, but clicking one navigates to the full Contests page instead of expanding. */
  function ctRenderPreview(data) {
    const host = document.getElementById("contests-preview-root");
    if (!host || !data) return "";
    const sig = JSON.stringify(data);
    if (host.dataset.sig === sig) return host.dataset.sig;
    host.dataset.sig = sig;

    const contests = data.map(parseContest).filter((c) => c.rows.length);
    const contested = shuffle(contests.filter((c) => c.contested));
    const rest = shuffle(contests.filter((c) => !c.contested));
    const picks = contested.concat(rest).slice(0, PREVIEW_MAX);

    const grid = document.createElement("div");
    grid.className = "ct-card-grid ct-preview-grid";
    picks.forEach((c) => {
      const card = makeCard(c);
      card.onclick = () => window.location.assign("/contests");
      grid.appendChild(card);
    });
    host.replaceChildren(grid);
    return String(sig.length);
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    contests: { render: ctRender, renderPreview: ctRenderPreview },
  });
})();
