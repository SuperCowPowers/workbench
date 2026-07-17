/* Example plugin clientside renderer.
 *
 * Ships in the plugin repo's assets/ folder. The workbench dashboard stages plugin
 * assets/ into its own assets tree (served at /assets/plugins/hello/render.js) and Dash
 * injects this <script> into every page head, so the "hello" namespace is available to
 * ClientsideFunction the same way the app's own assets are.
 *
 * plugin_page_assets.py wires a clientside callback:
 *   ClientsideFunction(namespace="hello", function_name="render")
 * Python fills a Store with data; this function owns the pixels (renders into #hello-root).
 */

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.hello = {
  render: function (data) {
    const root = document.getElementById("hello-root");
    if (!root) {
      return window.dash_clientside.no_update;
    }
    data = data || {};
    const items = data.items || [];

    const cards = items
      .map((it) => `<div class="hello-card">${it}</div>`)
      .join("");

    root.innerHTML = `
      <div class="hello-banner">${data.greeting || "Hello from a plugin asset!"}</div>
      <div class="hello-grid">${cards}</div>
    `;

    // Return a small status string into the (hidden) signal output
    return `rendered ${items.length} item(s)`;
  },
};
