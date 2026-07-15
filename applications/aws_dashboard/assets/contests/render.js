/* Model Contests page renderer.
 *
 * A Dash clientside callback (namespace "contests", function "render") calls
 * ctRender(data) whenever the server-side contest data changes. The data is a list
 * of contest reports (from Reports() /contests/*): each contest is a list of row
 * dicts — the champion row first, then the challengers ranked best-first, with
 * metric columns interleaved with Δ-vs-champion columns (positive = challenger
 * better). The renderer derives everything else (primary metric, contested flag)
 * from the rows.
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
  // contest_report(), multi-task already resolved). Name inference is only a
  // fallback for reports published before the column existed.
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
  function frameworkOf(row) {
    if (row.framework) return FW_KEY[row.framework] || "other";
    const n = (row.model || "").toLowerCase();
    for (const [pat, fw] of [["-mt", "mt"], ["chemprop", "chemprop"], ["pytorch", "pytorch"], ["xgb", "xgb"]]) {
      if (n.includes(pat)) return fw;
    }
    return "other";
  }

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
      primary,
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
  // Deltas at 2 decimal places; anything that rounds to 0.00 displays (and colors) as 0
  const zeroish = (v) => v == null || Math.abs(v) < 0.005;
  const fmtDelta = (v) =>
    v == null || Number.isNaN(v) ? "—" : zeroish(v) ? "0" : (v > 0 ? "+" : "") + Number(v).toFixed(2);

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

  /* Bar length encodes "goodness" of the primary metric across this contest's models:
     the best model gets the longest bar (RMSE inverts: lower is better). Bar color is
     the model's framework hue, so the ladder reads as a colored skyline. */
  function goodness(c) {
    const vals = c.rows.map((r) => r[c.primary]).filter((v) => v != null);
    const min = Math.min(...vals), max = Math.max(...vals), span = max - min;
    return (v) => {
      if (v == null || !isFinite(v)) return 0;
      const raw = span < 1e-12 ? 1 : c.primary === "rmse" ? (max - v) / span : (v - min) / span;
      return 0.1 + raw * 0.9; // floor so the worst model still shows a stub
    };
  }

  function ladderRow(row, value, delta, cls, good, marker) {
    const fw = frameworkOf(row);
    const side = delta != null && !zeroish(delta) && delta > 0 ? "pos" : "neg";
    return `<div class="ct-lrow ${cls}">
      ${marker}
      <span class="ct-lname" title="${row.model}">${row.model}</span>
      <span class="ct-vbar"><span class="ct-vbar-fill" style="width:${(good(value) * 100).toFixed(1)}%; background:var(--ct-f-${fw})"></span></span>
      <span class="ct-lval">${fmt(value)}</span>
      <span class="ct-ldelta ${side}">${cls === "champ" ? "" : fmtDelta(delta)}</span>
    </div>`;
  }

  function ladderHTML(c) {
    const dKey = "Δ" + c.primary;
    const good = goodness(c);
    const rows = [];
    if (c.champion) rows.push(ladderRow(c.champion, c.champion[c.primary], null, "champ", good, rankMarker("champion", 0)));
    c.challengers.slice(0, LADDER_MAX).forEach((r, i) => {
      rows.push(ladderRow(r, r[c.primary], r[dKey], "chall", good, rankMarker("challenger", i)));
    });
    const extra = c.challengers.length - LADDER_MAX;
    const more = extra > 0 ? `<div class="ct-more">+${extra} more challenger${extra > 1 ? "s" : ""}</div>` : "";
    return `<div class="ct-ladder"><div class="ct-lhead">
        <span></span><span></span><span></span><span>${c.primary}</span><span>Δ</span>
      </div>${rows.join("")}${more}</div>`;
  }

  // ---------- expanded table (full contest report) ----------
  function tableHTML(c) {
    const cols = Object.keys(c.rows[0]).filter((k) => !META_COLS.has(k));
    const head = `<tr><th></th><th class="ct-ta-l">model</th><th class="ct-ta-l">type</th>${cols
      .map((k) => `<th>${k}</th>`)
      .join("")}</tr>`;
    let challengerIdx = 0;
    const body = c.rows
      .map((r) => {
        const cells = cols
          .map((k) => {
            const isDelta = k.startsWith("Δ");
            const cls = isDelta ? (zeroish(r[k]) ? "ct-zero" : r[k] > 0 ? "ct-pos" : "ct-neg") : "";
            return `<td class="${cls}">${isDelta ? fmtDelta(r[k]) : fmt(r[k])}</td>`;
          })
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
      .map((fw) => `<span class="ct-legend-chip f-${fw}"><span class="ct-dot" style="background:var(--ct-f-${fw})"></span>${FRAMEWORK_LABEL[fw]}</span>`)
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

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    contests: { render: ctRender },
  });
})();
