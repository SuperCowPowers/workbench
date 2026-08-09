# Bootstrap and DBC Theme Hooks in Dash

## Theme Hooks Overview

### CSS Classes
Bootstrap provides pre-defined classes (`container`, `row`, `col`, `btn`, etc.) for responsive layouts. DBC components apply these automatically, and you can customize via the `className` property.

### CSS Variables
Bootstrap uses CSS variables for flexible theming:

- `--bs-body-bg`: Background color
- `--bs-body-color`: Text color
- `--bs-border-color`: Border color

These adjust automatically based on the `data-bs-theme` attribute.

### Data Attributes
The `data-bs-theme` attribute (`light` or `dark`) on a root container dynamically adjusts all Bootstrap variables globally.

---

## How Theme Switching Works

Workbench supports **instant theme switching**—no page reload required.

### What Happens on Theme Change

1. User clicks a theme in the settings menu
2. Clientside callback (JavaScript) executes:
   - Saves to `localStorage` and cookie (`wb_theme`)
   - Swaps Bootstrap stylesheet URL
   - Updates `data-bs-theme` attribute
   - Updates `workbench-theme-store` dcc.Store
3. Page-level callback fires, calling `set_theme()` on each plugin
4. Plugins re-render figures with new colors from ThemeManager

### Server-Side Theme Detection

The server resolves the theme from the `wb_theme` cookie on the request in flight, so viewers
on a shared dashboard each get their own theme:

```python
ThemeManager().current_theme()     # pin -> request cookie -> configured default
ThemeManager().current_template()  # that theme's Plotly template
```

Nothing is stored between requests. A single-theme process (a plugin unit test, an example
script) can call `set_theme("light")` to pin a theme that outranks any cookie; `set_theme("auto")`
returns to per-request resolution.

### Building Figures in a Plugin

Build figures with `self.theme_manager.figure()` rather than `go.Figure()`:

```python
fig = self.theme_manager.figure(data=traces)
```

A bare `go.Figure()` bakes in Plotly's process-global default template at construction. That
global is shared across request threads and can't follow the viewer, so those figures render in
the server's startup theme no matter who's looking. `figure()` takes the same arguments and
stamps the request's own template.

If a plugin caches a figure across requests, re-stamp it before returning:

```python
return self.theme_manager.apply_template(self.cached_figure)
```

---

## Two Main Theming Concepts

### CSS for Web Interface
Styles the overall layout, buttons, dropdowns, fonts, and background colors. Use Bootstrap or custom CSS files. Switch themes dynamically via CSS class changes.

### Plotly Templates for Figures
Styles Plotly figures (background, gridlines, colorscales, fonts). Use predefined templates or create custom JSON templates.

**Resource**: [dash-bootstrap-templates](https://github.com/AnnMarieW/dash-bootstrap-templates)
