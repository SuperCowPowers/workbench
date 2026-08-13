/* Model Comparison page renderer.
 *
 * A Dash clientside callback (namespace "mc", function "render") calls render(data,
 * endpoint, challenger) whenever the contest data or the selection changes. `data` is a
 * list of contests {endpoint, rows} (the full /contests/* reports, champion row first,
 * challengers ranked, metric columns interleaved with Δ-vs-champion columns).
 *
 * The renderer owns the left rail (#mc-rail), the header (#mc-head), and the expanded
 * comparison table (#mc-table) -- it draws straight into the DOM. Selection is pushed
 * back to Dash via set_props on two Stores (mc_selected_endpoint, mc_selected_challenger),
 * which drive the server-rendered plots. The plots themselves are Dash-owned; we only
 * write the model name into each plot row's header.
 */

(function () {
  const CROWN = '<svg viewBox="0 0 16 16" width="15" height="15" fill="currentColor"><path d="M1.5 5.5l3.2 2.6L8 3.5l3.3 4.6 3.2-2.6-1.2 7H2.7l-1.2-7z"/></svg>';
  const CHEVRON = '<svg viewBox="0 0 20 20" fill="none" width="15" height="15"><path d="M6 8l4 4 4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  // Groups the user has collapsed in the rail. Module-level so the state survives rail
  // rebuilds (the rail re-renders on every selection change). Empty = all open.
  const collapsedGroups = new Set();

  // Columns that are contest metadata rather than comparable metrics
  const META_COLS = new Set([
    "model", "role", "framework", "endpoint", "created", "inference_run", "timestamp", "contested",
  ]);

  // Framework -> the CSS color token suffix (--mc-f-*). Anything unlisted gets "other",
  // and is still labeled with its own name.
  const FW_COLOR = {
    chemprop: "chemprop", "chemprop-desc": "desc", "multi-task": "mt", xgboost: "xgb",
    pytorch: "pytorch", transformer: "transformer", sklearn: "sklearn", meta: "meta",
  };
  const colorOf = (framework) => FW_COLOR[framework] || "other";
  const dot = (framework) => `<span class="mc-dot" style="background:var(--mc-f-${colorOf(framework)})"></span>`;

  // Rank markers: gold crown (champion), then silver, bronze, open circles.
  const MEDALS = ["silver", "bronze"];
  function rankMarker(role, challengerIdx) {
    if (role === "champion") return `<span class="mc-crown">${CROWN}</span>`;
    const medal = MEDALS[challengerIdx];
    return `<span class="mc-medal ${medal ? "mc-medal-" + medal : "mc-medal-open"}"></span>`;
  }

  // Floats show 2 significant digits; integers (support counts) stay exact
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
    return d > 0 ? "mc-pos" : "mc-neg";
  };

  // The rail groups contests by their pipeline-hierarchy group (the top level of the group
  // path the view attaches -- the same grouping the ML Pipelines and Contests pages use).
  // Contests not in any pipeline fall through to "Other".
  const groupOf = (c) => (c.group && c.group[0]) || "Other";

  const firstRow = (c) => (c.rows && c.rows[0]) || {};
  const findRole = (c, role) => c.rows.find((r) => r.role === role) || null;
  const bestChallenger = (c) => { const r = findRole(c, "challenger"); return r ? r.model : null; };

  function setProps(id, value) {
    const dc = window.dash_clientside;
    if (dc && dc.set_props) dc.set_props(id, { data: value });
  }
  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  // The order the rail lists contests in: by group, then by endpoint. The default selection
  // reads the first entry, so it always matches the first item on screen.
  const railOrder = (data) =>
    data.slice().sort((a, b) => groupOf(a).localeCompare(groupOf(b)) || a.endpoint.localeCompare(b.endpoint));

  // ---------- left rail ----------
  function buildRail(data, endpoint) {
    const rail = document.getElementById("mc-rail");
    if (!rail) return;

    // Bucketing an already-ordered list leaves the Map in rail order, with each group's
    // contests already sorted.
    const groups = new Map();
    railOrder(data).forEach((c) => {
      const type = groupOf(c);
      if (!groups.has(type)) groups.set(type, []);
      groups.get(type).push(c);
    });

    const frag = document.createDocumentFragment();
    groups.forEach((contests, type) => {
      const sec = document.createElement("div");
      sec.className = "mc-group" + (collapsedGroups.has(type) ? " collapsed" : "");
      const h = document.createElement("div");
      h.className = "mc-group-h";
      h.innerHTML = `<span class="mc-caret">${CHEVRON}</span><span>${type}</span>`;
      h.onclick = () => {
        if (sec.classList.toggle("collapsed")) collapsedGroups.add(type);
        else collapsedGroups.delete(type);
      };
      sec.appendChild(h);
      contests.forEach((c) => {
        const champion = findRole(c, "champion") || firstRow(c);
        const item = document.createElement("div");
        item.className = "mc-item" + (c.endpoint === endpoint ? " sel" : "");
        item.innerHTML =
          dot(champion.framework) +
          `<span class="mc-item-name" title="${c.endpoint}">${c.endpoint}</span>` +
          (c.recent_change ? '<span class="mc-rbadge" title="recent change"></span>' : "") +
          (firstRow(c).contested ? '<span class="mc-cbadge" title="contested"></span>' : "");
        item.onclick = () => {
          setProps("mc_selected_endpoint", c.endpoint);
          setProps("mc_selected_challenger", bestChallenger(c));
        };
        sec.appendChild(item);
      });
      frag.appendChild(sec);
    });
    rail.replaceChildren(frag);
  }

  // ---------- framework legend (every framework present across every contest) ----------
  function buildLegend(data) {
    const el = document.getElementById("mc-legend");
    if (!el) return;
    const present = new Set();
    data.forEach((c) => c.rows.forEach((r) => r.framework && present.add(r.framework)));
    el.innerHTML = [...present]
      .sort()
      .map((fw) => `<span class="mc-legend-chip">${dot(fw)}${fw}</span>`)
      .join("");
  }

  // ---------- header ----------
  function buildHead(contest) {
    const head = document.getElementById("mc-head");
    if (!head) return;
    head.innerHTML =
      `<span class="mc-h-title">${contest.endpoint}</span>` +
      (contest.recent_change ? '<span class="mc-pill mc-pill-recent">recent change</span>' : "") +
      (firstRow(contest).contested ? '<span class="mc-pill mc-pill-contested">contested</span>' : "") +
      `<span class="mc-h-run">${firstRow(contest).inference_run || ""}</span>`;
  }

  // ---------- expanded comparison table ----------
  function buildTable(contest, challenger) {
    const el = document.getElementById("mc-table");
    if (!el) return;
    const cols = Object.keys(firstRow(contest)).filter((k) => !META_COLS.has(k) && !k.startsWith("Δ"));
    const head =
      '<tr><th></th><th class="mc-l">model</th><th class="mc-l">type</th>' +
      cols.map((k) => `<th>${k}</th>`).join("") + "</tr>";

    let challengerIdx = 0;
    const body = contest.rows
      .map((r) => {
        const champ = r.role === "champion";
        const marker = rankMarker(r.role, champ ? 0 : challengerIdx++);
        const cells = cols.map((k) => `<td class="${deltaClass(r, k, champ)}">${fmt(r[k])}</td>`).join("");
        const cls = champ ? "mc-champ-row" : "mc-chall-row" + (r.model === challenger ? " rsel" : "");
        const attr = champ ? "" : ` data-model="${r.model}"`;
        return `<tr class="${cls}"${attr}>
          <td class="mc-rank">${marker}</td>
          <td class="mc-l">${r.model}</td>
          <td class="mc-l mc-type">${dot(r.framework)}${r.framework || "other"}</td>
          ${cells}</tr>`;
      })
      .join("");
    el.innerHTML = `<div class="mc-table-wrap"><table class="mc-table">${head}${body}</table></div>`;

    el.querySelectorAll("tr.mc-chall-row").forEach((tr) => {
      tr.onclick = () => setProps("mc_selected_challenger", tr.getAttribute("data-model"));
    });
  }

  /* Render entrypoint. Draws the rail, header, and table, and labels the plot rows.
     If the selected endpoint is missing/invalid, falls back to the first contest in the rail
     and writes it back -- the set_props re-triggers render with a valid selection. */
  function render(data, endpoint, challenger) {
    if (!data || !data.length) return "";
    buildLegend(data);

    // Nothing selected yet means a fresh page load, so the deep link decides. Reading the
    // query string here rather than in a server callback keeps it race-free: the fallback
    // below would otherwise seed the first contest before the server's answer arrived.
    const wanted = endpoint || new URLSearchParams(window.location.search).get("name");

    // Fall back to the first contest in the rail when nothing is selected yet, or when the
    // URL named an endpoint that isn't published. Seeding a Store re-triggers render.
    const contest = data.find((c) => c.endpoint === wanted) || railOrder(data)[0];
    if (contest.endpoint !== endpoint) {
      buildRail(data, contest.endpoint);
      setProps("mc_selected_endpoint", contest.endpoint);
      setProps("mc_selected_challenger", bestChallenger(contest));
      return "init";
    }

    // A deep link names only the endpoint, so fill in the contest's best challenger. The
    // bestChallenger guard matters: without it a contest that has none would seed null
    // over null forever, and the table would never draw.
    if (!challenger && bestChallenger(contest)) {
      setProps("mc_selected_challenger", bestChallenger(contest));
      return "init";
    }

    // Mirror the selection into the query string. replaceState rather than a write to
    // dcc.Location: with refresh="callback-nav" that would be a navigation and would remount
    // the page on every click. Setting "name" leaves any other param intact.
    const params = new URLSearchParams(window.location.search);
    params.set("name", endpoint);
    const search = "?" + params.toString();
    if (search !== window.location.search) window.history.replaceState(null, "", search);

    buildRail(data, endpoint);
    buildHead(contest);
    buildTable(contest, challenger);
    setText("mc-champion-name", (findRole(contest, "champion") || {}).model || "");
    setText("mc-challenger-name", challenger || "(no challenger)");
    return `${endpoint}|${challenger}`;
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, { mc: { render: render } });
})();
